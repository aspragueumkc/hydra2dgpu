<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="structure_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="structure_type">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="1" name="Weir" type="int"/>
            <Option value="2" name="Culvert" type="int"/>
            <Option value="3" name="Gate" type="int"/>
            <Option value="4" name="Bridge" type="int"/>
            <Option value="5" name="Pump" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="crest_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="enabled">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="1" name="Yes" type="int"/>
            <Option value="0" name="No" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="width">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="height">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="diameter">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_shape">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_code">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="57" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_rise">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_span">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_area_m2">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_barrels">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="10" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_slope">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="inlet_invert_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="outlet_invert_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="entrance_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="exit_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="embankment_enabled">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="0" name="No" type="int"/>
            <Option value="1" name="Yes" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="embankment_crest_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="embankment_overflow_width">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="embankment_weir_coeff">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="length">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="roughness_n">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="coeff">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="cd">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="opening">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="q_pump">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="max_flow">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="inlet_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="outlet_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="stacked_enabled">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="0" name="No" type="int"/>
            <Option value="1" name="Yes" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="use_redistribution">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="0" name="No" type="int"/>
            <Option value="1" name="Yes" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="influence_width">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="upstream_buffer">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="downstream_buffer">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="deck_soffit_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="deck_top_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="model_top_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="under_layers">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="10" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="over_layers">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="10" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="pier_count">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="20" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="pier_width">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Structure ID"/>
    <alias index="1" name="Structure Type"/>
    <alias index="2" name="Crest Elevation"/>
    <alias index="3" name="Enabled"/>
    <alias index="4" name="Width"/>
    <alias index="5" name="Height"/>
    <alias index="6" name="Diameter"/>
    <alias index="7" name="Culvert Shape"/>
    <alias index="8" name="Culvert Code"/>
    <alias index="9" name="Culvert Rise"/>
    <alias index="10" name="Culvert Span"/>
    <alias index="11" name="Culvert Area"/>
    <alias index="12" name="Barrels"/>
    <alias index="13" name="Culvert Slope"/>
    <alias index="14" name="Inlet Invert Elev."/>
    <alias index="15" name="Outlet Invert Elev."/>
    <alias index="16" name="Entrance Loss K"/>
    <alias index="17" name="Exit Loss K"/>
    <alias index="18" name="Embankment Enabled"/>
    <alias index="19" name="Embankment Crest Elev."/>
    <alias index="20" name="Overflow Width"/>
    <alias index="21" name="Weir Coefficient"/>
    <alias index="22" name="Length"/>
    <alias index="23" name="Roughness n"/>
    <alias index="24" name="Coefficient"/>
    <alias index="25" name="Discharge Coeff."/>
    <alias index="26" name="Opening"/>
    <alias index="27" name="Pump Flow"/>
    <alias index="28" name="Max Flow"/>
    <alias index="29" name="Inlet Loss K"/>
    <alias index="30" name="Outlet Loss K"/>
    <alias index="31" name="Stacked Enabled"/>
    <alias index="32" name="Use Redistribution"/>
    <alias index="33" name="Influence Width"/>
    <alias index="34" name="Upstream Buffer"/>
    <alias index="35" name="Downstream Buffer"/>
    <alias index="36" name="Deck Soffit Elev."/>
    <alias index="37" name="Deck Top Elev."/>
    <alias index="38" name="Model Top Elev."/>
    <alias index="39" name="Under Layers"/>
    <alias index="40" name="Over Layers"/>
    <alias index="41" name="Pier Count"/>
    <alias index="42" name="Pier Width"/>
  </aliases>
  <defaults>
    <default expression="1" field="structure_type" applyOnUpdate="0"/>
    <default expression="0.0" field="crest_elev" applyOnUpdate="0"/>
    <default expression="1" field="enabled" applyOnUpdate="0"/>
    <default expression="0.035" field="roughness_n" applyOnUpdate="0"/>
    <default expression="30.0" field="length" applyOnUpdate="0"/>
    <default expression="0.5" field="entrance_loss_k" applyOnUpdate="0"/>
    <default expression="1.0" field="exit_loss_k" applyOnUpdate="0"/>
    <default expression="1" field="culvert_barrels" applyOnUpdate="0"/>
  </defaults>
  <constraintExpressions>
      <constraint exp="length(trim(&quot;structure_id&quot;)) &gt; 0" field="structure_id" desc=""/>
      <constraint exp="&quot;structure_type&quot; IN (1,2,3,4,5)" field="structure_type" desc=""/>
      <constraint exp="&quot;enabled&quot; IS NULL OR &quot;enabled&quot; IN (0,1)" field="enabled" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="structure_id"/>
    <field name="structure_type"/>
    <field name="crest_elev"/>
    <field name="enabled"/>
    <field name="width"/>
    <field name="height"/>
    <field name="diameter"/>
    <field name="culvert_shape"/>
    <field name="culvert_code"/>
    <field name="culvert_rise"/>
    <field name="culvert_span"/>
    <field name="culvert_area_m2"/>
    <field name="culvert_barrels"/>
    <field name="culvert_slope"/>
    <field name="inlet_invert_elev"/>
    <field name="outlet_invert_elev"/>
    <field name="entrance_loss_k"/>
    <field name="exit_loss_k"/>
    <field name="embankment_enabled"/>
    <field name="embankment_crest_elev"/>
    <field name="embankment_overflow_width"/>
    <field name="embankment_weir_coeff"/>
    <field name="length"/>
    <field name="roughness_n"/>
    <field name="coeff"/>
    <field name="cd"/>
    <field name="opening"/>
    <field name="q_pump"/>
    <field name="max_flow"/>
    <field name="inlet_loss_k"/>
    <field name="outlet_loss_k"/>
    <field name="stacked_enabled"/>
    <field name="use_redistribution"/>
    <field name="influence_width"/>
    <field name="upstream_buffer"/>
    <field name="downstream_buffer"/>
    <field name="deck_soffit_elev"/>
    <field name="deck_top_elev"/>
    <field name="model_top_elev"/>
    <field name="under_layers"/>
    <field name="over_layers"/>
    <field name="pier_count"/>
    <field name="pier_width"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>