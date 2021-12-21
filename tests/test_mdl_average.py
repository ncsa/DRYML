import pytest
from dryml import DryModelAverage, load_object
import objects


@pytest.mark.usefixtures("create_name")
def test_model_average_1(create_name):
    mdl_avg = DryModelAverage()
    mdl_avg.add_component(objects.HelloComponent(msg='Test'))
    mdl_avg.add_component(objects.HelloComponentB(msg='A'))

    mdl_avg.save_self(create_name)

    mdl_avg_2 = load_object(create_name)

    assert len(mdl_avg.components) == len(mdl_avg_2.components)

    assert mdl_avg.components[0].get_individual_hash() == \
        mdl_avg_2.components[0].get_individual_hash()
    assert mdl_avg.components[1].get_individual_hash() == \
        mdl_avg_2.components[1].get_individual_hash()
