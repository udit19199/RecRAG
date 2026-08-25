"""Guide extraction with fixed top kinds while leaving predicates open."""

from neo4j_graphrag.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType
from neo4j_graphrag.generation.prompts import ERExtractionTemplate

SCHEMA = GraphSchema(
    node_types=[
        NodeType(label="Person", description="A human.", properties=[PropertyType(name="name", type="STRING", description="Name."), PropertyType(name="gender", type="STRING", description="Male, female, or other identity as stated. Omit when not stated."), PropertyType(name="birth_date", type="STRING", description="Birth date as written."), PropertyType(name="death_date", type="STRING", description="Death date as written."), PropertyType(name="nationality", type="STRING", description="Nationality as written."), PropertyType(name="occupation", type="STRING", description="Job or role as written.")], additional_properties=True),
        NodeType(label="Organization", description="A group, firm, or school.", properties=[PropertyType(name="name", type="STRING", description="Name."), PropertyType(name="kind", type="STRING", description="Company, school, or group as written."), PropertyType(name="founded_on", type="STRING", description="Founding date as written."), PropertyType(name="headquarters", type="STRING", description="Head place as written.")], additional_properties=True),
        NodeType(label="Place", description="A city, country, or building.", properties=[PropertyType(name="name", type="STRING", description="Name."), PropertyType(name="kind", type="STRING", description="City, country, or building as written."), PropertyType(name="country", type="STRING", description="Country as written.")], additional_properties=True),
        NodeType(label="CreativeWork", description="A book, film, or song.", properties=[PropertyType(name="name", type="STRING", description="Name."), PropertyType(name="kind", type="STRING", description="Book, film, or song as written."), PropertyType(name="published_on", type="STRING", description="Publish date as written."), PropertyType(name="language", type="STRING", description="Language as written.")], additional_properties=True),
        NodeType(label="Event", description="A war, award ceremony, or election.", properties=[PropertyType(name="name", type="STRING", description="Name."), PropertyType(name="held_on", type="STRING", description="Event date as written."), PropertyType(name="location", type="STRING", description="Event place as written.")], additional_properties=True),
        NodeType(label="Concept", description="An award, field, genre, or language.", properties=[PropertyType(name="name", type="STRING", description="Name."), PropertyType(name="kind", type="STRING", description="Award, field, or genre as written.")], additional_properties=True),
        NodeType(label="Thing", description="Fallback when no narrower kind fits.", properties=[PropertyType(name="name", type="STRING", description="Name.")], additional_properties=True),
    ],
    relationship_types=[
        RelationshipType(label="BORN_IN", description="Person born in Place."),
        RelationshipType(label="DIED_IN", description="Person died in Place."),
        RelationshipType(label="LOCATED_IN", description="Inside a Place."),
        RelationshipType(label="PART_OF", description="Part of a whole."),
        RelationshipType(label="MEMBER_OF", description="Person in Organization."),
        RelationshipType(label="CREATED_BY", description="Work made by Person or Organization."),
        RelationshipType(label="SPOUSE_OF", description="Married to."),
        RelationshipType(label="CHILD_OF", description="Child of."),
        RelationshipType(label="AWARDED", description="Given award."),
        RelationshipType(label="HAS_NATIONALITY", description="Person linked to Place."),
    ],
    additional_node_types=True,
    additional_relationship_types=True,
)

EXTRACTION_PROMPT = ERExtractionTemplate.DEFAULT_TEMPLATE + (
    "\n\nRules:\n"
    "- Every node needs a name.\n"
    "- Fill a field only when text states it. Omit it when not stated.\n"
    "- Use Thing only when no narrower kind fits.\n"
    "- Link name is UPPER_SNAKE, short, from text.\n"
    "- One fact per link. No facts outside text.\n"
)
