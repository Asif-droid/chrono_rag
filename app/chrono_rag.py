"""
Chrono RAG: track how client requirements change over time, using Neo4j.

Graph model
-----------
(:Stakeholder)-[:SENT]->(:Message)-[:STATED]->(:Requirement)-[:ABOUT]->(:Feature)
(:Requirement)-[:SUPERSEDES]->(:Requirement)      new value replaces old one
(:Requirement)-[:CONFIRMS]->(:Requirement)        repeats an existing one
(:Requirement)-[:CANCELS]->(:Requirement)         withdraws an existing one
(:Requirement)-[:CONFLICTS_WITH]->(:Requirement)  needs human review

Requirement.status: active | superseded | cancelled | pending_review | duplicate | cancel_record

Setup
-----
pip install neo4j google-genai
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=...
export GEMINI_API_KEY=...
python chrono_rag.py
"""
import json
import os
from datetime import date

from neo4j import GraphDatabase
from google import genai
from google.genai import types

# ---------------------------------------------------------------- config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://chrono-neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")  # change via env var if you prefer another Gemini model

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment


def run(query, **params):
    with driver.session() as s:
        return [r.data() for r in s.run(query, **params)]


# ---------------------------------------------------------------- LLM helper
def call_llm_json(prompt: str) -> dict:
    """Ask the LLM for JSON only, strip code fences, parse."""
    resp = llm.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",  # Gemini JSON mode
            temperature=0,                          # deterministic extraction
        ),
    )
    text = (resp.text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()  # safety net
    return json.loads(text)


# ---------------------------------------------------------------- schema + load
def setup_schema():
    for label in ["Stakeholder", "Feature", "Message", "Requirement"]:
        run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")


def load_reference_data(stakeholders, features):
    for s in stakeholders:
        run("""MERGE (s:Stakeholder {id:$id})
               SET s.name=$name, s.role=$role, s.authority_level=$auth""",
            id=s["stakeholder_id"], name=s["name"], role=s["role"],
            auth=s["authority_level"])
    for f in features:
        run("""MERGE (f:Feature {id:$id})
               SET f.name=$name, f.aliases=$aliases""",
            id=f["feature_id"], name=f["canonical_name"],
            aliases=[a.lower() for a in f.get("aliases", [])])


# ---------------------------------------------------------------- step 1: extract
def extract_claims(msg, stakeholder, features, last_feature_name):
    feature_list = "\n".join(
        f"- {f['canonical_name']} (aliases: {', '.join(f.get('aliases', []))})"
        for f in features
    )
    prompt = f"""You extract requirement claims from a client message.

Known features:
{feature_list}

Message date: {msg['date']}
Sender: {stakeholder['name']} ({stakeholder['role']})
Last feature discussed by this sender: {last_feature_name or 'none'}
Message: "{msg['raw_text']}"

Return JSON only:
{{"claims": [{{
  "feature_mention": "<feature name as written, or the last feature if the message says 'it'/'that', or null>",
  "attribute": "<short snake_case aspect, e.g. primary_color, login_method>",
  "value": "<the requested value>",
  "effective_date": "<YYYY-MM-DD if the message says when it starts, else null>",
  "confidence": <0..1>
}}]}}
One message may contain several claims. Return an empty list if there are none."""
    return call_llm_json(prompt).get("claims", [])


# ---------------------------------------------------------------- step 2: resolve
def resolve_feature(mention, features):
    if not mention:
        return None
    m = mention.strip().lower()
    for f in features:
        names = [f["canonical_name"].lower()] + [a.lower() for a in f.get("aliases", [])]
        if m in names or any(n in m for n in names):
            return f
    return None  # TODO: embedding fallback, or push to a review queue


# ---------------------------------------------------------------- step 3: compare
def get_active_requirements(feature_id):
    return run("""
        MATCH (r:Requirement {status:'active'})-[:ABOUT]->(:Feature {id:$fid})
        RETURN r.id AS id, r.attribute AS attribute, r.value AS value,
               r.stated_at AS stated_at, r.authority AS authority
        ORDER BY r.stated_at""", fid=feature_id)


def classify_relation(claim, active, msg):
    existing = "\n".join(
        f"- {r['id']}: {r['attribute']} = {r['value']} (stated {r['stated_at']})"
        for r in active) or "(none)"
    prompt = f"""A new requirement claim arrived. Compare it with the active requirements.

Active requirements for this feature:
{existing}

New claim: attribute={claim['attribute']}, value={claim['value']}
Original message: "{msg['raw_text']}"

Classify as one of:
NEW      - not related to any active requirement
UPDATE   - replaces the value of an active requirement (give target_id)
CONFIRM  - repeats an active requirement (give target_id)
CANCEL   - withdraws an active requirement (give target_id)
REVERT   - restores an older value; target_id is the currently active requirement
CONFLICT - contradicts something but it is unclear whether it overrides

If related to an existing attribute, reuse that exact attribute name.
Return JSON only: {{"relation":"...", "target_id":"<id or null>", "attribute":"<attribute to store>"}}"""
    return call_llm_json(prompt)


# ---------------------------------------------------------------- step 4+5: apply + store
def create_requirement(req_id, feature_id, claim, attribute, msg, stakeholder, status):
    run("""
        MATCH (m:Message {id:$mid}), (f:Feature {id:$fid})
        CREATE (r:Requirement {
            id:$rid, attribute:$attr, value:$val, status:$status,
            stated_at:date($stated), effective_at:$eff,
            confidence:$conf, authority:$auth, stakeholder_id:$sid})
        CREATE (m)-[:STATED]->(r)
        CREATE (r)-[:ABOUT]->(f)""",
        mid=msg["message_id"], fid=feature_id, rid=req_id, attr=attribute,
        val=claim["value"], status=status, stated=msg["date"],
        eff=claim.get("effective_date"), conf=claim.get("confidence"),
        auth=stakeholder["authority_level"], sid=stakeholder["stakeholder_id"])


def link(new_id, rel, old_id):
    run(f"""MATCH (a:Requirement {{id:$a}}), (b:Requirement {{id:$b}})
            MERGE (a)-[:{rel}]->(b)""", a=new_id, b=old_id)


def set_status(req_id, status):
    run("MATCH (r:Requirement {id:$id}) SET r.status=$s", id=req_id, s=status)


def apply_decision(req_id, decision, claim, feature, msg, stakeholder, active):
    relation = decision["relation"].upper()
    target_id = decision.get("target_id")
    attribute = decision.get("attribute") or claim["attribute"]
    target = next((r for r in active if r["id"] == target_id), None)
    fid = feature["feature_id"]
    my_auth = stakeholder["authority_level"]

    if relation == "NEW" or (relation in ("UPDATE", "REVERT", "CANCEL", "CONFIRM") and not target):
        create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "active")

    elif relation == "CONFIRM":
        create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "duplicate")
        link(req_id, "CONFIRMS", target["id"])

    elif relation in ("UPDATE", "REVERT"):
        if my_auth >= target["authority"]:          # authority rule lives in CODE
            create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "active")
            link(req_id, "SUPERSEDES", target["id"])
            set_status(target["id"], "superseded")
        else:
            create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "pending_review")
            link(req_id, "CONFLICTS_WITH", target["id"])

    elif relation == "CANCEL":
        if my_auth >= target["authority"]:
            create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "cancel_record")
            link(req_id, "CANCELS", target["id"])
            set_status(target["id"], "cancelled")
        else:
            create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "pending_review")
            link(req_id, "CONFLICTS_WITH", target["id"])

    else:  # CONFLICT or unknown
        create_requirement(req_id, fid, claim, attribute, msg, stakeholder, "pending_review")
        if target:
            link(req_id, "CONFLICTS_WITH", target["id"])

    print(f"   {req_id}: {relation:8} {attribute} = {claim['value']}")


# ---------------------------------------------------------------- main loop
def process_messages(messages, stakeholders, features):
    sh_by_id = {s["stakeholder_id"]: s for s in stakeholders}
    last_feature = {}  # stakeholder_id -> last feature name, for "change it to blue"

    for msg in sorted(messages, key=lambda m: m["date"]):
        done = run("MATCH (m:Message {id:$id}) RETURN m.processed AS p", id=msg["message_id"])
        if done and done[0]["p"]:
            continue  # idempotent: skip already processed messages

        stakeholder = sh_by_id[msg["stakeholder_id"]]
        run("""MERGE (m:Message {id:$id})
               SET m.date=date($date), m.source_type=$src, m.raw_text=$txt
               WITH m MATCH (s:Stakeholder {id:$sid}) MERGE (s)-[:SENT]->(m)""",
            id=msg["message_id"], date=msg["date"], src=msg["source_type"],
            txt=msg["raw_text"], sid=msg["stakeholder_id"])

        print(f"{msg['message_id']} ({msg['date']}): {msg['raw_text']}")
        claims = extract_claims(msg, stakeholder, features, last_feature.get(msg["stakeholder_id"]))

        for i, claim in enumerate(claims, 1):
            feature = resolve_feature(claim.get("feature_mention"), features)
            if not feature:
                print(f"   ! unresolved feature '{claim.get('feature_mention')}', needs review")
                continue
            last_feature[msg["stakeholder_id"]] = feature["canonical_name"]

            active = get_active_requirements(feature["feature_id"])
            decision = classify_relation(claim, active, msg)
            req_id = f"R-{msg['message_id']}-{i}"
            apply_decision(req_id, decision, claim, feature, msg, stakeholder, active)

        run("MATCH (m:Message {id:$id}) SET m.processed=true", id=msg["message_id"])


# ---------------------------------------------------------------- queries (the "chrono RAG" read side)
def current_state(feature_id, as_of=None):
    """What is true now (or as of a date). Respects effective_at."""
    as_of = as_of or date.today().isoformat()
    return run("""
        MATCH (r:Requirement {status:'active'})-[:ABOUT]->(:Feature {id:$fid})
        WHERE r.effective_at IS NULL OR date(r.effective_at) <= date($as_of)
        RETURN r.attribute AS attribute, r.value AS value, r.stated_at AS since
        ORDER BY attribute""", fid=feature_id, as_of=as_of)


def timeline(feature_id, attribute=None):
    """Full history of a feature, oldest first, with what each entry replaced."""
    return run("""
        MATCH (r:Requirement)-[:ABOUT]->(:Feature {id:$fid})
        WHERE $attr IS NULL OR r.attribute = $attr
        OPTIONAL MATCH (r)-[rel:SUPERSEDES|CANCELS|CONFIRMS|CONFLICTS_WITH]->(old:Requirement)
        MATCH (m:Message)-[:STATED]->(r)
        RETURN r.attribute AS attribute, r.value AS value, r.status AS status,
               toString(r.stated_at) AS stated_at, type(rel) AS relation,
               old.value AS previous_value, m.id AS source_message, m.raw_text AS source_text
        ORDER BY r.stated_at, r.id""", fid=feature_id, attr=attribute)


def answer_question(question, feature_id, feature_name):
    """Retrieve the timeline from the graph, then let the LLM answer from it."""
    context = json.dumps({"current": current_state(feature_id),
                          "timeline": timeline(feature_id)}, indent=2, default=str)
    resp = llm.models.generate_content(
        model=MODEL,
        contents=(f"Feature: {feature_name}\nRequirement history (graph data):\n{context}\n\n"
                  f"Question: {question}\n"
                  "Answer only from this data. Cite message ids. "
                  "'current' is the source of truth for what is true now."),
        config=types.GenerateContentConfig(temperature=0),
    )
    return resp.text


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    with open("stakeholders.json") as f:
        stakeholders = json.load(f)["stakeholders"]
    with open("features.json") as f:
        features = json.load(f)["features"]
    with open("messages.json") as f:
        messages = json.load(f)["messages"]

    setup_schema()
    load_reference_data(stakeholders, features)
    process_messages(messages, stakeholders, features)

    print("\nCurrent state of Dashboard:")
    for row in current_state("F001"):
        print("  ", row)

    print("\nDashboard timeline:")
    for row in timeline("F001"):
        print("  ", row)

    print("\n" + answer_question("What color is the dashboard and how did it change?",
                                 "F001", "Dashboard"))
    driver.close()