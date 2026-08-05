# Devlog #13 — we worked ALOT (18 hours logged)

so this was the grind day. 18 hours of training babysitting and honestly it felt like 40.

the modal training kept dying every hour. the client would crash and take the whole run with it. we restarted like 3 times before figuring out `modal run -d` (detached) keeps it alive on their servers even if our computer shits the bed. tested it by literally killing the process and it kept going. ez fix but we worked ALOT to get there.

then we ran out of credits at 93% and switched accounts, but the other account "finished" the run in 6 MILLISECONDS which was sus af. turns out it just re-uploaded the old checkpoint and trained nothing. the resume logic had a bug where it thought 6200 >= 3799 meant done. patched it. another lesson learned.

also we made the model creative lol. added diversity sampling to the trajectory generator (5 solutions per task, keep the novel ones), a novelty bonus reward for RLVR, and a creativity rule in the self-awareness prompt. so the fat cat will be innovative, not just memorize.

current state: SFT is done at 91% (good enough), resume bug fixed, phase 2 (trajectory SFT with creativity + self-awareness baked in) is up next.

18 hours. we worked ALOT. the fat cat will be worth it. 😼
