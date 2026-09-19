import pandas as pd

from src.naive_values.contribution_scores_log import Responsibility
from src.util.analysis_functions import compute_handovers, compute_handovers_diff, jaccard_similarity, \
    compute_handovers_manual
from src.util.loader import load_event_log, load_ocel, flatten_log


def sum_values(order_dict, invoice_dict) -> dict:
    events = set().union(order_dict.keys(), invoice_dict.keys())
    return {e: (9/14 * order_dict.get(e, 0.0)) + (5/14 * invoice_dict.get(e, 0.0)) for e in events}


def compute_value(responsibility_calc, granularity='event', value="banzhaf"):
    if value == "banzhaf" or value == "shapley":
        val = responsibility_calc.compute_simple_values(granularity, value)
    else:
        val = responsibility_calc.compute_nested_value(granularity, value)
    return val

OCEL_PATH = "../../data/ocels/ocel_log_paper_example.json"
ocel = load_ocel(OCEL_PATH)

log = flatten_log(ocel, "Order", enrich_w_resource=True, resource_obj_type=["User"])
# analysis: handover graph
hw_net1 = set(compute_handovers_manual(log).keys())
analysis_func = lambda x: jaccard_similarity(hw_net1, set(compute_handovers_manual(x).keys()))


responsibility_calc = Responsibility(log, False,
                                       analysis_function=analysis_func,
                                       event_attr="ocel:eid")
order_shap = compute_value(responsibility_calc, granularity='event', value="shapley")
order_banz = compute_value(responsibility_calc, granularity='event', value="banzhaf")
order_owen = compute_value(responsibility_calc, granularity='event', value="owen")
order_bowen = compute_value(responsibility_calc, granularity='event', value="banzhaf-owen")

ocel = load_ocel(OCEL_PATH)
log = flatten_log(ocel, "Invoice", enrich_w_resource=True, resource_obj_type=["User"])

#analysis: handover graph
hw_net3 = set(compute_handovers_manual(log).keys())
analysis_func = lambda x: jaccard_similarity(hw_net3, set(compute_handovers_manual(x).keys()))

responsibility_calc = Responsibility(log, False,
                                       analysis_function=analysis_func,
                                       event_attr="ocel:eid")
inv_shap = compute_value(responsibility_calc, granularity='event', value="shapley")
inv_banz = compute_value(responsibility_calc, granularity='event', value="banzhaf")
inv_owen = compute_value(responsibility_calc, granularity='event', value="owen")
inv_bowen = compute_value(responsibility_calc, granularity='event', value="banzhaf-owen")

# aggregate Order + Invoice values per event (Shapley/Banzhaf additivity + null-player property)
shapley_total = sum_values(order_shap, inv_shap)
banzhaf_total = sum_values(order_banz, inv_banz)
owen_total = sum_values(order_owen, inv_owen)
banzhaf_owen_total = sum_values(order_bowen, inv_bowen)

#events = sorted(set(shapley_total) | set(banzhaf_total) | set(owen_total) | set(banzhaf_owen_total))
events = sorted(set(shapley_total) | set(owen_total) )
results_table = pd.DataFrame({
    #"Shapley (aggr)": [shapley_total.get(e, 0.0) for e in events],
    "Shapley (O)": [order_shap.get(e, 0.0) for e in events],
    #"Shapley (I)": [inv_shap.get(e, 0.0) for e in events],
    #"Banzhaf (aggr)": [banzhaf_total.get(e, 0.0) for e in events],
    "Banzhaf (O)": [order_banz.get(e, 0.0) for e in events],
    #"Banzhaf (I)": [inv_banz.get(e, 0.0) for e in events],
    #"Owen (aggr)": [owen_total.get(e, 0.0) for e in events],
    "Owen (O)": [order_owen.get(e, 0.0) for e in events],
    #"Owen (I)": [inv_owen.get(e, 0.0) for e in events],
    #"Banzhaf-Owen (aggr)": [banzhaf_owen_total.get(e, 0.0) for e in events],
    "Banzhaf-Owen (O)": [order_bowen.get(e, 0.0) for e in events],
    #"Banzhaf-Owen (I)": [inv_bowen.get(e, 0.0) for e in events],
}, index=events)
results_table.index.name = "event"

print(results_table.to_latex(float_format="%.2f"))

