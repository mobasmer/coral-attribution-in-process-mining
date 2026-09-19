import pm4py
from pm4py.ocel import ocel_flattening
from pm4py.objects.log.util.sampling import sample_log
from ocpa.objects.log.importer.ocel import factory


def load_ocel(path: str):
    return pm4py.read_ocel2(path)

def flatten_log(ocel, object_type: str,
                sample:int = None,
                enrich_w_resource:bool = False,
                resource_attr:str = "org:resource",
                resource_obj_type:list[str] = None):

    if enrich_w_resource:
        if resource_obj_type is None:
            resource_obj_type = ["employees", "customers"]

        rel = ocel.relations
        resource_links = rel[rel["ocel:type"].isin(resource_obj_type)][["ocel:eid", "ocel:oid"]]
        resource_links = resource_links.rename(columns={"ocel:oid": resource_attr})

        ocel.events = ocel.events.merge(resource_links, on="ocel:eid", how="left")
        ocel.events[resource_attr] = ocel.events[resource_attr].fillna("")

    log = ocel_flattening(ocel, object_type=object_type)

    if sample is not None:
        # sample N distinct values from the column
        sampled_values = log["case:concept:name"].drop_duplicates().sample(n=sample, random_state=26)

        # keep all rows where the column's value is in that sample
        log = log[log["case:concept:name"].isin(sampled_values)]

    return log

def load_event_log(path:str, sample:int = None):
    log = pm4py.read_xes(path)
    if sample is not None:
        log = sample_log(log, sample)

    return log

def load__with_resource_information(log, obj_type_for_resoure:str = "employees", attr_name:str = "resource:id"):
    return

def load_ocel_with_leading_type(path: str, object_type: str):
    try:
        parameters = {
            "execution_extraction": "leading_type",
            "leading_type": object_type
        }
        ocel = factory.apply(path, parameters=parameters)

    except Exception:
        print("Log not found")
        raise Exception

    return ocel

def build_log_from_subset(log, is_ocel:bool = False, events:list[str] = None, objects:list[str] = None, event_id_attr:str = "EventId"):
    assert events is not None or objects is not None

    if is_ocel:
        if objects is not None:
            subset_log = pm4py.filtering.filter_ocel_objects(log, objects, positive=True)

            if events is not None:
                subset_log = pm4py.filtering.filter_ocel_events(subset_log, events, positive=False)
                # accounting for Owen values mapping, in this case events representing complement of subset, i.e., removing those events
        elif events is not None:
            subset_log = pm4py.filtering.filter_ocel_events(log, events, positive=True)
    else:
        if objects is not None:
            subset_log = pm4py.filter_event_attribute_values(
                log,
                'case:concept:name',
                objects,
                level='case',
                retain=True
            )

            if events is not None: # accounting for Owen values mapping, in this case events representing complement of subset, i.e., removing those events
                subset_log = pm4py.filter_event_attribute_values(
                                        subset_log,
                                        event_id_attr,
                                        events,
                                        level='event',
                                        retain=False
                                    )
        elif events is not None:
            subset_log = pm4py.filter_event_attribute_values(
                                        log,
                                        event_id_attr,
                                        events,
                                        level='event',
                                        retain=True
                                    )
    return subset_log