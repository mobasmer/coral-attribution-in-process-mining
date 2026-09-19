import operator
import pm4py

def compute_dfg(contexts, events, relation, query_type):
    dfg = set()
    dfg_values = dict()
    for e1, e2, c, val in relation:
        if c in contexts:
            if e1 in events and e2 in events:
                edge = (events[e1], events[e2])
                dfg.add(edge)
                if query_type != "boolean":
                    match query_type:
                        case "max":
                            dfg_values[edge] = max(dfg_values[edge], val) if edge in dfg_values else val
                        case "min":
                            dfg_values[edge] = min(dfg_values[edge], val) if edge in dfg_values else val
                        case "sum":
                            dfg_values[edge] = dfg_values[edge] + val if edge in dfg_values else val

    return dfg, dfg_values

# handover analysis
def jaccard_similarity(set1:set, set2:set):
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if len(intersection) == 0 and len(union) == 0:
        return 1.0
    else:
        return len(intersection) / len(union)

def compute_handovers(log, k:int = 0):
    # todo: adapt to handover computation over oc dfg to align with example?
    try:
        hw_net = pm4py.discover_handover_of_work_network(log)
    except ValueError:
        return set()

    if k == 0:
        return set(hw_net.connections.keys())
    else:
        top_k_handovers = sorted(hw_net.connections.items(), key=operator.itemgetter(1))[:k]
        return set([k for k, v in top_k_handovers])

def compute_handovers_manual(log, resource_attr:str = "org:resource", case_id_attr:str = "case:concept:name",
                              timestamp_attr:str = "time:timestamp", k:int = 0):
    # manual handover-of-work computation: within each context (case), look at the
    # directly-follows relation over time and record a handover whenever two
    # consecutive events are performed by different resources, tallying frequency
    connections: dict = dict()

    for c in log[case_id_attr].unique():
        trace = log[log[case_id_attr] == c].sort_values(timestamp_attr)[resource_attr].tolist()

        for r1, r2 in zip(trace, trace[1:]):
            if r1 != r2 and r1 != "" and r2 != "":
                edge = (r1, r2)
                connections[edge] = connections.get(edge, 0) + 1

    if k == 0:
        return connections
    else:
        return dict(sorted(connections.items(), key=operator.itemgetter(1))[:k])


def compute_handovers_diff(hw_net1, hw_net2):
    jacc_sim = jaccard_similarity(hw_net1, hw_net2)
    return jacc_sim

def compute_oc_handovers(ocel, object_groups:list, resource_obj_type:list = None, k:int = 0):
    # object-centric handover network: for every object-centric case notion (a group of
    # correlated objects), take each object's own trace (its related events, ordered by
    # time) and record a handover whenever two consecutive events in that trace are
    # linked to different resource objects (objects of type in resource_obj_type)
    if resource_obj_type is None:
        resource_obj_type = ["employees", "customers"]

    relations = ocel.relations
    eid_col = ocel.event_id_column
    oid_col = ocel.object_id_column
    type_col = ocel.object_type_column
    ts_col = ocel.event_timestamp

    resource_relations = relations[relations[type_col].isin(resource_obj_type)]
    event_to_resources = resource_relations.groupby(eid_col)[oid_col].apply(set).to_dict()

    connections: dict = dict()

    for group in object_groups:
        for oid in group:
            trace = relations[relations[oid_col] == oid].sort_values(ts_col)[eid_col].tolist()

            for e1, e2 in zip(trace, trace[1:]):
                for r1 in event_to_resources.get(e1, set()):
                    for r2 in event_to_resources.get(e2, set()):
                        if r1 != r2:
                            edge = (r1, r2)
                            connections[edge] = connections.get(edge, 0) + 1

    if k == 0:
        return set(connections.keys())
    else:
        top_k_handovers = sorted(connections.items(), key=operator.itemgetter(1))[:k]
        return set([edge for edge, _ in top_k_handovers])

# efg analysis
def compute_efg(log):
    efg = pm4py.discover_eventually_follows_graph(log)
    return efg.keys()

def compute_efg_metric(log, num_activities):
    efg_edges = compute_efg(log)

    return len(efg_edges) / (num_activities * num_activities)

def compute_conformance_checking(log, activity_a:str, activity_b:str, constraint_type="Response", eventid_attr:str = "EventId"):
    violations = 0
    satisfactions = 0
    violation_details = []

    if constraint_type == "Response":
        for c in log["case:concept:name"].unique():
            df = log[log["case:concept:name"] == c]

            events = df[df["concept:name"].isin([activity_a, activity_b])] \
                .sort_values("time:timestamp")

            seen_b = False
            for _, row in events.iloc[::-1].iterrows():
                act = row["concept:name"]

                if act == activity_b:
                    seen_b = True
                elif act == activity_a:
                    if seen_b:
                        satisfactions += 1
                    else:
                        violations += 1
                        violation_details.append((c, "no B after this A", row["time:timestamp"]))
    elif constraint_type == "AlternateResponse":
        violations = 0
        violation_details = []

        for c in log["case:concept:name"].unique():
            df = log[log["case:concept:name"] == c]

            # keep only A/B events, sorted by time
            events = df[df["concept:name"].isin([activity_a, activity_b])] \
                .sort_values("time:timestamp")

            expecting_b = False  # True after an A that hasn't been matched by a B yet
            pending_a_time = None

            for _, row in events.iterrows():
                act = row["concept:name"]

                if act == activity_a:
                    if expecting_b:
                        # a new A occurred before the previous A got its B -> violation
                        violations += 1
                        violation_details.append((c, "unmatched A", pending_a_time))
                    expecting_b = True
                    pending_a_time = row["time:timestamp"]

                elif act == activity_b:
                    if expecting_b:
                        expecting_b = False  # this B satisfies the pending A

            # end of case: if still waiting for a B, that last A is a violation
            if expecting_b:
                violations += 1
                violation_details.append((c, "unmatched A at end of case", pending_a_time))
    else:
        pass

    return violations