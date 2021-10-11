base = exec

# General functions
a1 = $(word 1,$(subst _, , $(1)))
a2 = $(word 2,$(subst _, , $(1)))
a3 = $(word 3,$(subst _, , $(1)))
a4 = $(word 4,$(subst _, , $(1)))
a5 = $(word 5,$(subst _, , $(1)))
a6 = $(word 6,$(subst _, , $(1)))
a7 = $(word 7,$(subst _, , $(1)))
a8 = $(word 8,$(subst _, , $(1)))

function_list = michalewicz2_ michalewicz5_ michalewicz10_ hartmann_ ackley_ rosenbrock_ shekel_
model_list = dkp_ #gp_ #deepgp_ gidgp_
learn_noise = lnt_ lnf_
retrain = rtt_ rtf_
run = 0 1 2 3 4 5 6 7 8 9

f_m = $(foreach pre,$(function_list),$(addprefix $(pre),$(model_list)))
f_m_l = $(foreach pre,$(f_m),$(addprefix $(pre),$(learn_noise)))
f_m_l_r = $(foreach pre,$(f_m_l),$(addprefix $(pre),$(retrain)))
f_m_l_r_run = $(foreach pre,$(f_m_l_r),$(addprefix $(pre),$(run)))

results/%: experiment.py
	$(base) python $< $@ --function $(call a1,$*) --model $(call a2,$*) --$(call a3,$*) --$(call a4,$*) --run $(call a5,$*)
experiment: $(addprefix results/,$(f_m_l_r_run))

random_search = rs_

f_rs = $(foreach pre,$(function_list),$(addprefix $(pre),$(random_search)))
f_rs_run = $(foreach pre,$(f_rs),$(addprefix $(pre),$(run)))

results_rs/%: experiment.py
	$(base) python $< $@ --function $(call a1,$*) --model $(call a2,$*) --run $(call a3,$*)
rs_experiment: $(addprefix results_rs/,$(f_rs_run))
