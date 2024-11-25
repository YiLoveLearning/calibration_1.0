import pandas as pd


bh_map = {
    'aao_influent_1_1_q': ('influent_q1_handled', 12),
    'aao_influent_1_1_tcod': ('influent_tcod_handled', 1),
    'aao_influent_1_1_tp': ('influent_tp_handled', 1),
    'aao_influent_1_1_tn': ('influent_tkn_handled', 1),
    'aao_influent_1_1_frsnhx_tkn': ('influent_frsnhx_tkn_handled', 1),
    'aao_influent_1_1_t': ('influent_t', 1),
    # 没有carbon和pac
    'aao_flowdivider3_1_1_influx_q': ('internalreflux1_1', 24),
    'aao_clarifier_1_1_sludge_target_q': ('clarifierunderflow1', 12),
    'aao_flowdivider4_1_1_sludge_q': ('excesssludge1', 12),
    'aao_cstr3_1_1_qair_ntp': ('airflow1_1_1', 24),
    'aao_cstr4_1_1_qair_ntp': ('airflow1_1_1', 24),
    'aao_cstr5_1_1_qair_ntp': ('airflow1_1_2', 24),
    'aao_cstr6_1_1_qair_ntp': ('airflow1_1_2', 24),
    'aao_cstr7_1_1_qair_ntp': ('airflow1_1_3', 24),

    'aao_influent_1_2_q': ('influent_q1_handled', 12),
    'aao_influent_1_2_tcod': ('influent_tcod_handled', 1),
    'aao_influent_1_2_tp': ('influent_tp_handled', 1),
    'aao_influent_1_2_tn': ('influent_tkn_handled', 1),
    'aao_influent_1_2_frsnhx_tkn': ('influent_frsnhx_tkn_handled', 1),
    'aao_influent_1_2_t': ('influent_t', 1),
    # 没有carbon和pac
    'aao_flowdivider3_1_2_influx_q': ('internalreflux1_2', 24),
    'aao_clarifier_1_2_sludge_target_q': ('clarifierunderflow1', 12),
    'aao_flowdivider4_1_2_sludge_q': ('excesssludge1', 12),
    'aao_cstr3_1_2_qair_ntp': ('airflow1_2_1', 24),
    'aao_cstr4_1_2_qair_ntp': ('airflow1_2_1', 24),
    'aao_cstr5_1_2_qair_ntp': ('airflow1_2_2', 24),
    'aao_cstr6_1_2_qair_ntp': ('airflow1_2_2', 24),
    'aao_cstr7_1_2_qair_ntp': ('airflow1_2_3', 24),


    'aao_influent_2_1_q': ('influent_q2_handled', 12),
    'aao_influent_2_1_tcod': ('influent_tcod_handled', 1),
    'aao_influent_2_1_tp': ('influent_tp_handled', 1),
    'aao_influent_2_1_tn': ('influent_tkn_handled', 1),
    'aao_influent_2_1_frsnhx_tkn': ('influent_frsnhx_tkn_handled', 1),
    'aao_influent_2_1_t': ('influent_t', 1),
    # 没有carbon和pac
    'aao_flowdivider3_2_1_influx_q': ('internalreflux2_1', 24),
    'aao_clarifier_2_1_sludge_target_q': ('clarifierunderflow2', 12),
    'aao_flowdivider4_2_1_sludge_q': ('excesssludge2', 12),
    'aao_cstr3_2_1_qair_ntp': ('airflow2_1_1', 24),
    'aao_cstr4_2_1_qair_ntp': ('airflow2_1_1', 24),
    'aao_cstr5_2_1_qair_ntp': ('airflow2_1_2', 24),
    'aao_cstr6_2_1_qair_ntp': ('airflow2_1_2', 24),
    'aao_cstr7_2_1_qair_ntp': ('airflow2_1_3', 24),

    'aao_influent_2_2_q': ('influent_q2_handled', 12),
    'aao_influent_2_2_tcod': ('influent_tcod_handled', 1),
    'aao_influent_2_2_tp': ('influent_tp_handled', 1),
    'aao_influent_2_2_tn': ('influent_tkn_handled', 1),
    'aao_influent_2_2_frsnhx_tkn': ('influent_frsnhx_tkn_handled', 1),
    'aao_influent_2_2_t': ('influent_t', 1),
    # 没有carbon和pac
    'aao_flowdivider3_2_2_influx_q': ('internalreflux2_2', 24),
    'aao_clarifier_2_2_sludge_target_q': ('clarifierunderflow2', 12),
    'aao_flowdivider4_2_2_sludge_q': ('excesssludge2', 12),
    'aao_cstr3_2_2_qair_ntp': ('airflow2_2_1', 24),
    'aao_cstr4_2_2_qair_ntp': ('airflow2_2_1', 24),
    'aao_cstr5_2_2_qair_ntp': ('airflow2_2_2', 24),
    'aao_cstr6_2_2_qair_ntp': ('airflow2_2_2', 24),
    'aao_cstr7_2_2_qair_ntp': ('airflow2_2_3', 24),
}

def f1():
    cleaning_data = pd.read_excel("../../data/OfflineData/2024-08-27-2024-09-23_cleaning_data.xlsx", 
                                  parse_dates=['sampling_time'])

    for bh, (handled_bh, interval) in bh_map.items():
        cleaning_data[bh] = cleaning_data[handled_bh] * interval

    cleaning_data.to_excel("../../data/OfflineData/2024-08-27-2024-09-23_model_cleaning_data.xlsx", index=False)

f1()