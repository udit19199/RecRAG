from neo4j import Driver


def read_graph_statistics(
    driver: Driver,
    *,
    database: str,
) -> dict[str, int]:
    result = driver.execute_query(
        """
        CALL {
            MATCH (entity:__Entity__)
            RETURN count(entity) AS entity_count,
                   coalesce(
                       sum(CASE WHEN NOT (entity)--() THEN 1 ELSE 0 END), 0
                   ) AS isolated_entity_count
        }
        CALL {
            MATCH (entity:__Entity__)
            WITH entity.name AS name, count(*) AS name_count
            WHERE name_count > 1
            RETURN count(name) AS duplicate_entity_name_groups,
                   coalesce(sum(name_count), 0) AS duplicate_entity_nodes
        }
        CALL {
            MATCH (subject:__Entity__)-[relation]->(object:__Entity__)
            RETURN count(relation) AS relationship_count,
                   count(DISTINCT type(relation)) AS relation_type_count,
                   coalesce(
                       sum(CASE WHEN subject = object THEN 1 ELSE 0 END), 0
                   ) AS self_loop_count
        }
        RETURN entity_count,
               isolated_entity_count,
               duplicate_entity_name_groups,
               duplicate_entity_nodes,
               relationship_count,
               relation_type_count,
               self_loop_count
        """,
        database_=database,
    )
    row = result.records[0]
    return {
        key: int(row[key])
        for key in (
            "entity_count",
            "isolated_entity_count",
            "duplicate_entity_name_groups",
            "duplicate_entity_nodes",
            "relationship_count",
            "relation_type_count",
            "self_loop_count",
        )
    }
