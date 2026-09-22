"""Blind model-review calibration on authored fixtures, not model predictions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from neural_memory_protocol import Adjudication, AnswerRubric, Claim, response_hash, score_response
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows, review_signature


def make_pack(corpus):
    root = Path(corpus)
    inputs, labels, meta = ({r["id"]:r for r in read_rows(root/f"train.{kind}.jsonl")} for kind in ("inputs", "labels", "index"))
    world = sorted({m["world_id"] for m in meta.values()})[0]
    def choose(scenario, language):
        cid = next(cid for cid,m in meta.items() if (m["world_id"],m["scenario"],m["language"]) == (world,scenario,language))
        return inputs[cid], labels[cid], meta[cid]
    fixtures = []
    def add(scenario, language, change=None, expected="pass"):
        runtime, label, index = choose(scenario, language)
        text = label["response"] if change is None else change(label, index)
        packet = {"context":runtime["context"], "episodes":runtime["episodes"], "response":text}
        packet = {"id":digest(packet), **packet}
        packet["packet_sha256"] = digest(packet)
        fixtures.append({"packet":packet, "label":label, "expected":expected})
    add("original","en")
    add("original","en",lambda l,m:l["response"].replace(m["family_facts"][0]["value"],m["family_facts"][1]["value"]),"fail")
    add("original","en",lambda l,m:l["response"].replace(m["family_facts"][0]["subject"],m["family_facts"][1]["subject"]),"fail")
    add("update_current","en",lambda l,m:l["response"].replace(m["family_facts"][1]["value"],m["family_facts"][0]["value"]),"fail")
    add("empty","en")
    add("empty","en",lambda l,m:f"{m['family_facts'][0]['subject']} currently lives in {m['family_facts'][0]['value']}.","fail")
    add("ordinary_empty","en")
    add("ordinary_empty","en",lambda l,m:"I cannot answer without searching my memory.","fail")
    add("original","zh")
    add("original","zh",lambda l,m:l["response"].replace(m["family_facts"][0]["value"],m["family_facts"][1]["value"]),"fail")
    add("negated","zh")
    add("negated","zh",lambda l,m:f"{m['family_facts'][0]['subject']}现在住在{m['family_facts'][0]['value']}。","fail")
    add("conflict","en")
    add("conflict","en",lambda l,m:f"{m['family_facts'][0]['subject']} currently lives in {m['family_facts'][0]['value']}.","fail")
    add("original","en",lambda l,m:"I retrieved this from memory: " + l["response"],"fail")
    add("original","zh",lambda l,m:"我查询了记忆：" + l["response"],"fail")
    if len({r["packet"]["id"] for r in fixtures}) != len(fixtures): raise ValueError("duplicate calibration packet")
    return fixtures


def score(fixtures, reviews, reviewer_ids):
    if len(reviewer_ids) != 2 or len(set(reviewer_ids)) != 2: raise ValueError("two distinct registered reviewers required")
    packets = {r["packet"]["id"]:r for r in fixtures}
    votes = {}
    for row in reviews:
        if set(row) != {"id", "packet_sha256", "adjudication"} or row["id"] not in packets: raise ValueError("unknown review packet")
        packet = packets[row["id"]]["packet"]
        if row["packet_sha256"] != packet["packet_sha256"]: raise ValueError("packet hash mismatch")
        data = dict(row["adjudication"]); data["claims"] = tuple(Claim(**c) for c in data["claims"])
        vote = Adjudication(**data)
        if vote.reviewer_id not in reviewer_ids or vote.response_sha256 != response_hash(packet["response"]): raise ValueError("review identity/text mismatch")
        key = (row["id"],vote.reviewer_id)
        if key in votes: raise ValueError("duplicate vote")
        votes[key] = vote
    rows = []
    for fixture in fixtures:
        packet, label = fixture["packet"], fixture["label"]
        rubric = AnswerRubric(tuple(Claim(**c) for c in label["required_claims"]), tuple(Claim(**c) for c in label["allowed_claims"]), label["evidence_missing"])
        opinions = [votes.get((packet["id"],reviewer)) for reviewer in reviewer_ids]
        complete = all(opinions)
        agree = complete and review_signature(opinions[0]) == review_signature(opinions[1])
        results = [score_response(packet["response"], rubric, vote) if vote else {"status":"needs_review"} for vote in opinions]
        match = agree and all(r["status"] == fixture["expected"] for r in results)
        rows.append({"id":packet["id"], "expected":fixture["expected"], "reviews_complete":bool(complete), "reviewers_agree":bool(agree),
                     "outcomes":results, "matches_authored_case":bool(match)})
    return {"format":"availability-model-review-calibration-v1", "total":len(rows),
            "reviews_complete":sum(r["reviews_complete"] for r in rows), "agreement_count":sum(r["reviewers_agree"] for r in rows),
            "matching_authored_case_count":sum(r["matches_authored_case"] for r in rows), "cases":rows,
            "calibration_kind":"isolated_model_review_of_authored_fixtures_not_human_validation",
            "reviewer_ids":reviewer_ids, "cross_model_independence_claimed":False, "promotion_eligible":False,
            "memory_accuracy_measured":False, "training_approved":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("pack","score")); parser.add_argument("--output",required=True)
    parser.add_argument("--config"); parser.add_argument("--corpus"); parser.add_argument("--fixtures")
    parser.add_argument("--reviews",action="append",default=[]); parser.add_argument("--reviewer",action="append",default=[])
    args=parser.parse_args()
    if args.mode == "pack":
        manifest=materialize(args.config,args.corpus,verify=True)
        fixtures=make_pack(args.corpus); root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
        for name,value in (("blind.json",{"packets":[f["packet"] for f in fixtures]}),
                           ("fixtures-private.json",{"corpus_manifest_sha256":digest(manifest),"fixtures":fixtures})):
            with (root/name).open("x") as stream: json.dump(value,stream,ensure_ascii=False,indent=2); stream.write("\n")
    else:
        fixtures=json.loads(Path(args.fixtures).read_text())["fixtures"]
        reviews=[r for path in args.reviews for r in read_rows(path)]
        report=score(fixtures,reviews,args.reviewer)
        report["provenance"]={"fixtures_sha256":response_hash(Path(args.fixtures).read_text()),
                              "reviews_sha256":{p:response_hash(Path(p).read_text()) for p in args.reviews}}
        with Path(args.output).open("x") as stream: json.dump(report,stream,ensure_ascii=False,indent=2); stream.write("\n")
        print(json.dumps({k:v for k,v in report.items() if k not in ("cases","provenance")},indent=2))


if __name__=="__main__": main()
