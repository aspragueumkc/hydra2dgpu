<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.9-Solothurn" styleCategories="Forms">
  <fieldConfiguration>
    <field name="fid">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="structure_id">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="List" name="map">
              <Option type="Map">
                <Option type="QString" name="test" value="1"/>
              </Option>
              <Option type="Map">
                <Option type="QString" name="can" value="2"/>
              </Option>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="structure_type">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="Map" name="map">
              <Option type="int" name="Bridge" value="4"/>
              <Option type="int" name="Culvert" value="2"/>
              <Option type="int" name="Gate" value="3"/>
              <Option type="int" name="Pump" value="5"/>
              <Option type="int" name="Weir" value="1"/>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="crest_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="enabled">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="width">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="height">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="diameter">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="culvert_shape">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="Map" name="map">
              <Option type="QString" name="Box" value="box"/>
              <Option type="QString" name="Circular" value="circular"/>
              <Option type="QString" name="Rectangular" value="rectangular"/>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="culvert_code">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="culvert_rise">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="culvert_span">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="culvert_area_m2">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="culvert_barrels">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="culvert_slope">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="inlet_invert_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="outlet_invert_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="entrance_loss_k">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="exit_loss_k">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="embankment_enabled">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="Map" name="map">
              <Option type="int" name="No" value="0"/>
              <Option type="int" name="Yes" value="1"/>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="embankment_crest_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="embankment_overflow_width">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="embankment_weir_coeff">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="length">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="roughness_n">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="coeff">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="cd">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="opening">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="q_pump">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="max_flow">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="inlet_loss_k">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="outlet_loss_k">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="stacked_enabled">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="use_redistribution">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="influence_width">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="upstream_buffer">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="downstream_buffer">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="deck_soffit_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="deck_top_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="model_top_elev">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="under_layers">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="over_layers">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="pier_count">
      <editWidget type="Range">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="pier_width">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <editform tolerant="1"></editform>
  <editforminit/>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[# -*- coding: utf-8 -*-
"""
QGIS forms can have a Python function that is called when the form is
opened.

Use this function to add extra logic to your forms.

Enter the name of the function in the "Python Init function"
field.
An example follows:
"""
from qgis.PyQt.QtWidgets import QWidget

def my_form_open(dialog, layer, feature):
    geom = feature.geometry()
    control = dialog.findChild(QWidget, "MyLineEdit")
]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <attributeEditorForm>
    <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
      <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
    </labelStyle>
    <attributeEditorField name="fid" horizontalStretch="0" index="0" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="structure_id" horizontalStretch="0" index="1" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Noto Sans,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="structure_type" horizontalStretch="0" index="2" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="crest_elev" horizontalStretch="0" index="3" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="enabled" horizontalStretch="0" index="4" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="width" horizontalStretch="0" index="5" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="height" horizontalStretch="0" index="6" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="diameter" horizontalStretch="0" index="7" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_shape" horizontalStretch="0" index="8" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_code" horizontalStretch="0" index="9" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_rise" horizontalStretch="0" index="10" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_span" horizontalStretch="0" index="11" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_area_m2" horizontalStretch="0" index="12" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_barrels" horizontalStretch="0" index="13" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="culvert_slope" horizontalStretch="0" index="14" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="inlet_invert_elev" horizontalStretch="0" index="15" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="outlet_invert_elev" horizontalStretch="0" index="16" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="entrance_loss_k" horizontalStretch="0" index="17" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="exit_loss_k" horizontalStretch="0" index="18" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="embankment_enabled" horizontalStretch="0" index="19" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="embankment_crest_elev" horizontalStretch="0" index="20" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="embankment_overflow_width" horizontalStretch="0" index="21" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="embankment_weir_coeff" horizontalStretch="0" index="22" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="length" horizontalStretch="0" index="23" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="roughness_n" horizontalStretch="0" index="24" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="coeff" horizontalStretch="0" index="25" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="cd" horizontalStretch="0" index="26" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="opening" horizontalStretch="0" index="27" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="q_pump" horizontalStretch="0" index="28" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="max_flow" horizontalStretch="0" index="29" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="inlet_loss_k" horizontalStretch="0" index="30" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="outlet_loss_k" horizontalStretch="0" index="31" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="stacked_enabled" horizontalStretch="0" index="32" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="use_redistribution" horizontalStretch="0" index="33" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="influence_width" horizontalStretch="0" index="34" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="upstream_buffer" horizontalStretch="0" index="35" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="downstream_buffer" horizontalStretch="0" index="36" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="deck_soffit_elev" horizontalStretch="0" index="37" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="deck_top_elev" horizontalStretch="0" index="38" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="model_top_elev" horizontalStretch="0" index="39" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="under_layers" horizontalStretch="0" index="40" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="over_layers" horizontalStretch="0" index="41" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="pier_count" horizontalStretch="0" index="42" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
    <attributeEditorField name="pier_width" horizontalStretch="0" index="43" verticalStretch="0" showLabel="1">
      <labelStyle overrideLabelFont="0" overrideLabelColor="0" labelColor="">
        <labelFont underline="0" italic="0" strikethrough="0" style="" bold="0" description="Sans Serif,9,-1,5,50,0,0,0,0,0"/>
      </labelStyle>
    </attributeEditorField>
  </attributeEditorForm>
  <editable>
    <field name="cd" editable="1"/>
    <field name="coeff" editable="1"/>
    <field name="crest_elev" editable="1"/>
    <field name="culvert_area_m2" editable="1"/>
    <field name="culvert_barrels" editable="1"/>
    <field name="culvert_code" editable="1"/>
    <field name="culvert_rise" editable="1"/>
    <field name="culvert_shape" editable="1"/>
    <field name="culvert_slope" editable="1"/>
    <field name="culvert_span" editable="1"/>
    <field name="deck_soffit_elev" editable="1"/>
    <field name="deck_top_elev" editable="1"/>
    <field name="diameter" editable="1"/>
    <field name="downstream_buffer" editable="1"/>
    <field name="embankment_crest_elev" editable="1"/>
    <field name="embankment_enabled" editable="1"/>
    <field name="embankment_overflow_width" editable="1"/>
    <field name="embankment_weir_coeff" editable="1"/>
    <field name="enabled" editable="1"/>
    <field name="entrance_loss_k" editable="1"/>
    <field name="exit_loss_k" editable="1"/>
    <field name="fid" editable="1"/>
    <field name="height" editable="1"/>
    <field name="influence_width" editable="1"/>
    <field name="inlet_invert_elev" editable="1"/>
    <field name="inlet_loss_k" editable="1"/>
    <field name="length" editable="1"/>
    <field name="max_flow" editable="1"/>
    <field name="model_top_elev" editable="1"/>
    <field name="opening" editable="1"/>
    <field name="outlet_invert_elev" editable="1"/>
    <field name="outlet_loss_k" editable="1"/>
    <field name="over_layers" editable="1"/>
    <field name="pier_count" editable="1"/>
    <field name="pier_width" editable="1"/>
    <field name="q_pump" editable="1"/>
    <field name="roughness_n" editable="1"/>
    <field name="stacked_enabled" editable="1"/>
    <field name="structure_id" editable="1"/>
    <field name="structure_type" editable="1"/>
    <field name="under_layers" editable="1"/>
    <field name="upstream_buffer" editable="1"/>
    <field name="use_redistribution" editable="1"/>
    <field name="width" editable="1"/>
  </editable>
  <labelOnTop>
    <field labelOnTop="0" name="cd"/>
    <field labelOnTop="0" name="coeff"/>
    <field labelOnTop="0" name="crest_elev"/>
    <field labelOnTop="0" name="culvert_area_m2"/>
    <field labelOnTop="0" name="culvert_barrels"/>
    <field labelOnTop="0" name="culvert_code"/>
    <field labelOnTop="0" name="culvert_rise"/>
    <field labelOnTop="0" name="culvert_shape"/>
    <field labelOnTop="0" name="culvert_slope"/>
    <field labelOnTop="0" name="culvert_span"/>
    <field labelOnTop="0" name="deck_soffit_elev"/>
    <field labelOnTop="0" name="deck_top_elev"/>
    <field labelOnTop="0" name="diameter"/>
    <field labelOnTop="0" name="downstream_buffer"/>
    <field labelOnTop="0" name="embankment_crest_elev"/>
    <field labelOnTop="0" name="embankment_enabled"/>
    <field labelOnTop="0" name="embankment_overflow_width"/>
    <field labelOnTop="0" name="embankment_weir_coeff"/>
    <field labelOnTop="0" name="enabled"/>
    <field labelOnTop="0" name="entrance_loss_k"/>
    <field labelOnTop="0" name="exit_loss_k"/>
    <field labelOnTop="0" name="fid"/>
    <field labelOnTop="0" name="height"/>
    <field labelOnTop="0" name="influence_width"/>
    <field labelOnTop="0" name="inlet_invert_elev"/>
    <field labelOnTop="0" name="inlet_loss_k"/>
    <field labelOnTop="0" name="length"/>
    <field labelOnTop="0" name="max_flow"/>
    <field labelOnTop="0" name="model_top_elev"/>
    <field labelOnTop="0" name="opening"/>
    <field labelOnTop="0" name="outlet_invert_elev"/>
    <field labelOnTop="0" name="outlet_loss_k"/>
    <field labelOnTop="0" name="over_layers"/>
    <field labelOnTop="0" name="pier_count"/>
    <field labelOnTop="0" name="pier_width"/>
    <field labelOnTop="0" name="q_pump"/>
    <field labelOnTop="0" name="roughness_n"/>
    <field labelOnTop="0" name="stacked_enabled"/>
    <field labelOnTop="0" name="structure_id"/>
    <field labelOnTop="0" name="structure_type"/>
    <field labelOnTop="0" name="under_layers"/>
    <field labelOnTop="0" name="upstream_buffer"/>
    <field labelOnTop="0" name="use_redistribution"/>
    <field labelOnTop="0" name="width"/>
  </labelOnTop>
  <reuseLastValue>
    <field name="cd" reuseLastValue="0"/>
    <field name="coeff" reuseLastValue="0"/>
    <field name="crest_elev" reuseLastValue="0"/>
    <field name="culvert_area_m2" reuseLastValue="0"/>
    <field name="culvert_barrels" reuseLastValue="0"/>
    <field name="culvert_code" reuseLastValue="0"/>
    <field name="culvert_rise" reuseLastValue="0"/>
    <field name="culvert_shape" reuseLastValue="0"/>
    <field name="culvert_slope" reuseLastValue="0"/>
    <field name="culvert_span" reuseLastValue="0"/>
    <field name="deck_soffit_elev" reuseLastValue="0"/>
    <field name="deck_top_elev" reuseLastValue="0"/>
    <field name="diameter" reuseLastValue="0"/>
    <field name="downstream_buffer" reuseLastValue="0"/>
    <field name="embankment_crest_elev" reuseLastValue="0"/>
    <field name="embankment_enabled" reuseLastValue="0"/>
    <field name="embankment_overflow_width" reuseLastValue="0"/>
    <field name="embankment_weir_coeff" reuseLastValue="0"/>
    <field name="enabled" reuseLastValue="0"/>
    <field name="entrance_loss_k" reuseLastValue="0"/>
    <field name="exit_loss_k" reuseLastValue="0"/>
    <field name="fid" reuseLastValue="0"/>
    <field name="height" reuseLastValue="0"/>
    <field name="influence_width" reuseLastValue="0"/>
    <field name="inlet_invert_elev" reuseLastValue="0"/>
    <field name="inlet_loss_k" reuseLastValue="0"/>
    <field name="length" reuseLastValue="0"/>
    <field name="max_flow" reuseLastValue="0"/>
    <field name="model_top_elev" reuseLastValue="0"/>
    <field name="opening" reuseLastValue="0"/>
    <field name="outlet_invert_elev" reuseLastValue="0"/>
    <field name="outlet_loss_k" reuseLastValue="0"/>
    <field name="over_layers" reuseLastValue="0"/>
    <field name="pier_count" reuseLastValue="0"/>
    <field name="pier_width" reuseLastValue="0"/>
    <field name="q_pump" reuseLastValue="0"/>
    <field name="roughness_n" reuseLastValue="0"/>
    <field name="stacked_enabled" reuseLastValue="0"/>
    <field name="structure_id" reuseLastValue="0"/>
    <field name="structure_type" reuseLastValue="0"/>
    <field name="under_layers" reuseLastValue="0"/>
    <field name="upstream_buffer" reuseLastValue="0"/>
    <field name="use_redistribution" reuseLastValue="0"/>
    <field name="width" reuseLastValue="0"/>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
  <layerGeometryType>1</layerGeometryType>
</qgis>
