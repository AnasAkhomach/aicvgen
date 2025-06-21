from src.models.data_models import (
    Item,
    MetadataModel,
    ContentItem,
    Subsection,
    StructuredCV,
    SkillEntry,
    ExperienceEntry,
    WorkflowState,
    AgentIO,
)


def test_metadata_model_fields():
    meta = MetadataModel(company="TestCorp", position="Engineer", extra={"foo": "bar"})
    assert meta.company == "TestCorp"
    assert meta.position == "Engineer"
    assert meta.extra["foo"] == "bar"


def test_item_metadata_type():
    item = Item(content="Test", metadata=MetadataModel(company="A"))
    assert isinstance(item.metadata, MetadataModel)


def test_contentitem_metadata_type():
    citem = ContentItem(
        content_type="test", original_content="x", metadata=MetadataModel(position="B")
    )
    assert isinstance(citem.metadata, MetadataModel)


def test_subsection_metadata_type():
    sub = Subsection(name="Sub", metadata=MetadataModel(location="C"))
    assert isinstance(sub.metadata, MetadataModel)


def test_structuredcv_metadata_type():
    cv = StructuredCV(metadata=MetadataModel(company="D"))
    assert isinstance(cv.metadata, MetadataModel)


def test_skillentry_metadata_type():
    skill = SkillEntry(name="Python", metadata=MetadataModel(extra={"level": "expert"}))
    assert isinstance(skill.metadata, MetadataModel)


def test_experienceentry_metadata_type():
    exp = ExperienceEntry(
        company="E",
        position="Dev",
        duration="1y",
        description="desc",
        metadata=MetadataModel(company="E"),
    )
    assert isinstance(exp.metadata, MetadataModel)


def test_workflowstate_metadata_type():
    state = WorkflowState(metadata=MetadataModel(company="F"))
    assert isinstance(state.metadata, MetadataModel)


def test_agentio_metadata_type():
    agentio = AgentIO(description="desc", metadata=MetadataModel(company="G"))
    assert isinstance(agentio.metadata, MetadataModel)
