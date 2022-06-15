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

########################## Branin experiment ################################
model_list = deepgp_
learn_noise = lnt_ #lnf_
retrain = rtt_1_ rtt_2_ rtt_5 rtt_10_ rtf_1_
norm = normt_ normf_
run = 0 1 2 3 4 5 6 7 8 9

b_m = $(addprefix branin_,$(model_list))
b_m_l = $(foreach pre,$(b_m),$(addprefix $(pre),$(learn_noise)))
b_m_l_r = $(foreach pre,$(b_m_l),$(addprefix $(pre),$(retrain)))
b_m_l_r_n = $(foreach pre,$(b_m_l_r),$(addprefix $(pre),$(norm)))
b_m_l_r_n_run = $(foreach pre,$(b_m_l_r_n),$(addprefix $(pre),$(run)))

results_rtfreq/%: experiment.py
	$(base) python $< $@ --exp_name rtfreq --function branin --model $(call a2,$*) --$(call a3,$*) --$(call a4,$*) --rt_every $(call a5,$*) --$(call a6,$*) --run $(call a7,$*)
branin_exp: $(addprefix results_rtfreq/,$(b_m_l_r_n_run))


########################## Other functions exp ##############################
function_list = dgpmich2_ dgpmich5_ dgpack2_ dgpack5_ #michalewicz5_ ackley_
model_list = sgpr_ #deepgp_ gidgp_
learn_noise = lnf_ # lnt_
retrain = rtt_5_ #rtt_10_ rtf_0_ #rtt_1_ rtt_2_ rtt_5_ rtt_10_ rtf_1_
norm = normf_ #normt_
run = 0 1 2 3 4 5 6 7 8 9

f_m = $(foreach pre,$(function_list),$(addprefix $(pre),$(model_list)))
f_m_l = $(foreach pre,$(f_m),$(addprefix $(pre),$(learn_noise)))
f_m_l_r = $(foreach pre,$(f_m_l),$(addprefix $(pre),$(retrain)))
f_m_l_r_n = $(foreach pre,$(f_m_l_r),$(addprefix $(pre),$(norm)))
f_m_l_r_n_run = $(foreach pre,$(f_m_l_r_n),$(addprefix $(pre),$(run)))

results_rtfreq/%: experiment.py
	$(base) python $< $@ --exp_name rtfreq --function $(call a1,$*) --model $(call a2,$*) --$(call a3,$*) --$(call a4,$*) --rt_every $(call a5,$*) --$(call a6,$*) --run $(call a7,$*)
experiment: $(addprefix results_rtfreq/,$(f_m_l_r_n_run))

random_search = rs_

f_rs = $(foreach pre,$(function_list),$(addprefix $(pre),$(random_search)))
f_rs_run = $(foreach pre,$(f_rs),$(addprefix $(pre),$(run)))

results_rs/%: experiment.py
	$(base) python $< $@ --exp_name rs --function $(call a1,$*) --model $(call a2,$*) --run $(call a3,$*)
rs_experiment: $(addprefix results_rs/,$(f_rs_run))


########################## Large scale exp ##############################
function_list = noisymich5_ noisyackley5_ noisyshekel_ noisyhart6_
model_list = deepgp_
num_inducing = 500_ #100_ 250_ 500_
run = 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29

f_m = $(foreach pre,$(function_list),$(addprefix $(pre),$(model_list)))
f_m_i = $(foreach pre,$(f_m),$(addprefix $(pre),$(num_inducing)))
f_m_l_run = $(foreach pre,$(f_m_i),$(addprefix $(pre),$(run)))

results_largescale/%: experiment.py
	$(base) python $< $@ --exp_name largescale --function $(call a1,$*) --model $(call a2,$*) --lnt --rtt --epochs 5000 --rt_every 1 --normf --num_query 100 --num_inducing $(call a3,$*) --fix_ips_t --run $(call a4,$*)
ls_experiment: $(addprefix results_largescale/,$(f_m_l_run))

f_rs = $(foreach pre,$(function_list),$(addprefix $(pre),$(random_search)))
f_rs_run = $(foreach pre,$(f_rs),$(addprefix $(pre),$(run)))

results_ls_rs/%: experiment.py
	$(base) python $< $@ --exp_name ls_rs --function $(call a1,$*) --model $(call a2,$*) --num_query 100 --run $(call a3,$*)
ls_rs_experiment: $(addprefix results_ls_rs/,$(f_rs_run))

