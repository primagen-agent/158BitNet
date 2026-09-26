"""EC-007 Phase A: paraphrase cache via the 3B backbone (plain-mode server).

For every (fact) and (query intent) in the JB-005 world inventory, ask the
3B model for N natural-dialogue rewrites. Binding is carried by a
programmatically attached "{person}: " prefix; the city must survive
verbatim (Phase B teacher-verifies and drops failures).

Cache: build/neural-memory-ec007/paraphrases.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from eval_locomo_neural import Server  # noqa: E402

OUT = ROOT / 'build/neural-memory-ec007/paraphrases.json'

EP_VARIANTS_EN = [
    'Rewrite as a casual chat message spoken by {p}. Keep the name {c} exactly, do not change it. Sentence: {p} currently {v} in {c}.',
    'Make this sound like natural spoken dialogue from {p}, keeping the city {c} unchanged: {p} currently {v} in {c}.',
    'Imagine {p} is texting a friend about life in {c}. Write one short natural message mentioning {c}.',
]
EP_VARIANTS_ZH = [
    '把这句话改写成{p}说的一句随意聊天消息，城市名{c}必须原样保留：{p}现在{v}{c}。',
    '想象{p}在和朋友闲聊时提到{c}，用自然的口语写一句，{c}不能改动。',
    '以{p}的口吻写一条关于{c}的日常消息，保留{c}原名。',
]
Q_VARIANTS_EN = [
    'Write one natural short question asking where {p} {v} now.',
    'How would a friend casually ask about the city {p} {v} in? One question.',
    'Phrase a natural question about where {p} {v} these days.',
]
Q_VARIANTS_ZH = [
    '用自然的口语写一句询问{p}现在{v}哪里的问句。',
    '朋友间随意地问{p}{v}在哪个城市，写一句问话。',
    '写一句自然的问题，问{p}目前{v}的地方。',
]


class Args:
    server = str(ROOT / 'build/c_neural_server')
    gguf = str(ROOT / 'models/bitcpm4-3b-tq2_0.gguf')
    memory_model = ''


def clean(text):
    t = text.strip().strip('"').strip('"“”').strip()
    return t.split('\n')[0][:220]


def main():
    config = json.loads((ROOT / 'training/memory/neural-system/data/JB-005.json').read_text())
    cache_path = ROOT / 'build/neural-memory-ec007/paraphrases.json'
    cache_path.parent.mkdir(exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    work_dir = ROOT / 'build/neural-memory-ec007'
    server = Server(Args, work_dir)
    server.cmd = [Args.server, Args.gguf]   # plain mode: no --memory-model
    server.start()
    jobs = []  # (key, prompt)
    people, cities = config['people'], config['cities']
    zh_verbs = {'home_city': ('住在', '住'), 'work_city': ('工作在', '工作')}
    en_verbs = {'home_city': ('lives', 'lives'), 'work_city': ('works', 'works')}
    # every person appears in one world with both relations; person index i ->
    # world i//2, relations cycled; mirror the dialogue generator's inventory
    for i, p in enumerate(people):
        for r_idx, relation in enumerate(('home_city', 'work_city')):
            city = cities[(i // 2) * 2 + r_idx]
            for lang, ep_v, q_v in (('en', EP_VARIANTS_EN, Q_VARIANTS_EN),
                                    ('zh', EP_VARIANTS_ZH, Q_VARIANTS_ZH)):
                verb = (en_verbs if lang == 'en' else zh_verbs)[relation][0]
                for v_i, tpl in enumerate(ep_v):
                    jobs.append((f'ep:{p}:{relation}:{lang}:{v_i}',
                                 tpl.format(p=p, c=city, v=verb)))
                qverb = (en_verbs if lang == 'en' else zh_verbs)[relation][1]
                for v_i, tpl in enumerate(q_v):
                    jobs.append((f'q:{p}:{relation}:{lang}:{v_i}',
                                 tpl.format(p=p, v=qverb)))

    todo = [(k, prm) for k, prm in jobs if k not in cache]
    print(f'{len(jobs)} jobs, {len(todo)} to generate', flush=True)
    t0 = time.time()
    for n, (key, prompt) in enumerate(todo):
        reply, _ = server.message(prompt, timeout=120)
        cache[key] = clean(reply)
        if (n + 1) % 25 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            rate = (n + 1) / (time.time() - t0)
            print(f'  {n+1}/{len(todo)} ({rate:.2f}/s, eta {((len(todo)-n-1)/rate)/60:.0f}m)',
                  flush=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    server.quit()
    print(f'cache written: {len(cache)} entries', flush=True)


if __name__ == '__main__':
    main()
