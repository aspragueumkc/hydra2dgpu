<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="structure_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="structure_type">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="1" name="Weir" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="2" name="Culvert" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="3" name="Gate" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="4" name="Bridge" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="5" name="Pump" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="crest_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="enabled">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="1" name="Yes" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="0" name="No" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="width">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="height">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="diameter">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_shape">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_code">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="57" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_rise">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_span">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_area_m2">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_barrels">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="1" name="Min" type="int"/>
              <Option value="10" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_slope">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="inlet_invert_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="outlet_invert_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="entrance_loss_k">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="exit_loss_k">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="embankment_enabled">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="0" name="No" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="1" name="Yes" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="embankment_crest_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="embankment_overflow_width">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="embankment_weir_coeff">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="length">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="roughness_n">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="coeff">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="cd">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="opening">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="q_pump">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="max_flow">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="inlet_loss_k">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="outlet_loss_k">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="stacked_enabled">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="0" name="No" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="1" name="Yes" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="use_redistribution">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="0" name="No" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="1" name="Yes" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="influence_width">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="upstream_buffer">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="downstream_buffer">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="deck_soffit_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="deck_top_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="model_top_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="under_layers">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="10" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="over_layers">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="10" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="pier_count">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="20" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="pier_width">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" field="" name=""/>
    <alias index="1" field="" name=""/>
    <alias index="2" field="" name=""/>
    <alias index="3" field="" name=""/>
    <alias index="4" field="" name=""/>
    <alias index="5" field="" name=""/>
    <alias index="6" field="" name=""/>
    <alias index="7" field="" name=""/>
    <alias index="8" field="" name=""/>
    <alias index="9" field="" name=""/>
    <alias index="10" field="" name=""/>
    <alias index="11" field="" name=""/>
    <alias index="12" field="" name=""/>
    <alias index="13" field="" name=""/>
    <alias index="14" field="" name=""/>
    <alias index="15" field="" name=""/>
    <alias index="16" field="" name=""/>
    <alias index="17" field="" name=""/>
    <alias index="18" field="" name=""/>
    <alias index="19" field="" name=""/>
    <alias index="20" field="" name=""/>
    <alias index="21" field="" name=""/>
    <alias index="22" field="" name=""/>
    <alias index="23" field="" name=""/>
    <alias index="24" field="" name=""/>
    <alias index="25" field="" name=""/>
    <alias index="26" field="" name=""/>
    <alias index="27" field="" name=""/>
    <alias index="28" field="" name=""/>
    <alias index="29" field="" name=""/>
    <alias index="30" field="" name=""/>
    <alias index="31" field="" name=""/>
    <alias index="32" field="" name=""/>
    <alias index="33" field="" name=""/>
    <alias index="34" field="" name=""/>
    <alias index="35" field="" name=""/>
    <alias index="36" field="" name=""/>
    <alias index="37" field="" name=""/>
    <alias index="38" field="" name=""/>
    <alias index="39" field="" name=""/>
    <alias index="40" field="" name=""/>
    <alias index="41" field="" name=""/>
    <alias index="42" field="" name=""/>
  </aliases>
  <defaults>
    <default expression="" field="structure_id" applyOnUpdate="0"/>
    <default expression="1" field="structure_type" applyOnUpdate="0"/>
    <default expression="0.0" field="crest_elev" applyOnUpdate="0"/>
    <default expression="1" field="enabled" applyOnUpdate="0"/>
    <default expression="" field="width" applyOnUpdate="0"/>
    <default expression="" field="height" applyOnUpdate="0"/>
    <default expression="" field="diameter" applyOnUpdate="0"/>
    <default expression="" field="culvert_shape" applyOnUpdate="0"/>
    <default expression="" field="culvert_code" applyOnUpdate="0"/>
    <default expression="" field="culvert_rise" applyOnUpdate="0"/>
    <default expression="" field="culvert_span" applyOnUpdate="0"/>
    <default expression="" field="culvert_area_m2" applyOnUpdate="0"/>
    <default expression="1" field="culvert_barrels" applyOnUpdate="0"/>
    <default expression="" field="culvert_slope" applyOnUpdate="0"/>
    <default expression="" field="inlet_invert_elev" applyOnUpdate="0"/>
    <default expression="" field="outlet_invert_elev" applyOnUpdate="0"/>
    <default expression="0.5" field="entrance_loss_k" applyOnUpdate="0"/>
    <default expression="1.0" field="exit_loss_k" applyOnUpdate="0"/>
    <default expression="" field="embankment_enabled" applyOnUpdate="0"/>
    <default expression="" field="embankment_crest_elev" applyOnUpdate="0"/>
    <default expression="" field="embankment_overflow_width" applyOnUpdate="0"/>
    <default expression="" field="embankment_weir_coeff" applyOnUpdate="0"/>
    <default expression="30.0" field="length" applyOnUpdate="0"/>
    <default expression="0.035" field="roughness_n" applyOnUpdate="0"/>
    <default expression="" field="coeff" applyOnUpdate="0"/>
    <default expression="" field="cd" applyOnUpdate="0"/>
    <default expression="" field="opening" applyOnUpdate="0"/>
    <default expression="" field="q_pump" applyOnUpdate="0"/>
    <default expression="" field="max_flow" applyOnUpdate="0"/>
    <default expression="" field="inlet_loss_k" applyOnUpdate="0"/>
    <default expression="" field="outlet_loss_k" applyOnUpdate="0"/>
    <default expression="" field="stacked_enabled" applyOnUpdate="0"/>
    <default expression="" field="use_redistribution" applyOnUpdate="0"/>
    <default expression="" field="influence_width" applyOnUpdate="0"/>
    <default expression="" field="upstream_buffer" applyOnUpdate="0"/>
    <default expression="" field="downstream_buffer" applyOnUpdate="0"/>
    <default expression="" field="deck_soffit_elev" applyOnUpdate="0"/>
    <default expression="" field="deck_top_elev" applyOnUpdate="0"/>
    <default expression="" field="model_top_elev" applyOnUpdate="0"/>
    <default expression="" field="under_layers" applyOnUpdate="0"/>
    <default expression="" field="over_layers" applyOnUpdate="0"/>
    <default expression="" field="pier_count" applyOnUpdate="0"/>
    <default expression="" field="pier_width" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="structure_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="structure_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="crest_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="enabled" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="width" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="height" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="diameter" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_shape" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_code" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_rise" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_span" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_area_m2" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_barrels" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="culvert_slope" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="inlet_invert_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="outlet_invert_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="entrance_loss_k" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="exit_loss_k" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="embankment_enabled" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="embankment_crest_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="embankment_overflow_width" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="embankment_weir_coeff" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="length" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="roughness_n" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="coeff" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="cd" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="opening" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="q_pump" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="max_flow" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="inlet_loss_k" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="outlet_loss_k" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="stacked_enabled" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="use_redistribution" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="influence_width" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="upstream_buffer" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="downstream_buffer" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="deck_soffit_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="deck_top_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="model_top_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="under_layers" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="over_layers" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="pier_count" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="pier_width" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;structure_id&quot;)) &gt; 0" field="structure_id" desc=""/>
    <constraint exp="&quot;structure_type&quot; IN (1,2,3,4,5)" field="structure_type" desc=""/>
    <constraint exp="" field="crest_elev" desc=""/>
    <constraint exp="&quot;enabled&quot; IS NULL OR &quot;enabled&quot; IN (0,1)" field="enabled" desc=""/>
    <constraint exp="" field="width" desc=""/>
    <constraint exp="" field="height" desc=""/>
    <constraint exp="" field="diameter" desc=""/>
    <constraint exp="" field="culvert_shape" desc=""/>
    <constraint exp="" field="culvert_code" desc=""/>
    <constraint exp="" field="culvert_rise" desc=""/>
    <constraint exp="" field="culvert_span" desc=""/>
    <constraint exp="" field="culvert_area_m2" desc=""/>
    <constraint exp="" field="culvert_barrels" desc=""/>
    <constraint exp="" field="culvert_slope" desc=""/>
    <constraint exp="" field="inlet_invert_elev" desc=""/>
    <constraint exp="" field="outlet_invert_elev" desc=""/>
    <constraint exp="" field="entrance_loss_k" desc=""/>
    <constraint exp="" field="exit_loss_k" desc=""/>
    <constraint exp="" field="embankment_enabled" desc=""/>
    <constraint exp="" field="embankment_crest_elev" desc=""/>
    <constraint exp="" field="embankment_overflow_width" desc=""/>
    <constraint exp="" field="embankment_weir_coeff" desc=""/>
    <constraint exp="" field="length" desc=""/>
    <constraint exp="" field="roughness_n" desc=""/>
    <constraint exp="" field="coeff" desc=""/>
    <constraint exp="" field="cd" desc=""/>
    <constraint exp="" field="opening" desc=""/>
    <constraint exp="" field="q_pump" desc=""/>
    <constraint exp="" field="max_flow" desc=""/>
    <constraint exp="" field="inlet_loss_k" desc=""/>
    <constraint exp="" field="outlet_loss_k" desc=""/>
    <constraint exp="" field="stacked_enabled" desc=""/>
    <constraint exp="" field="use_redistribution" desc=""/>
    <constraint exp="" field="influence_width" desc=""/>
    <constraint exp="" field="upstream_buffer" desc=""/>
    <constraint exp="" field="downstream_buffer" desc=""/>
    <constraint exp="" field="deck_soffit_elev" desc=""/>
    <constraint exp="" field="deck_top_elev" desc=""/>
    <constraint exp="" field="model_top_elev" desc=""/>
    <constraint exp="" field="under_layers" desc=""/>
    <constraint exp="" field="over_layers" desc=""/>
    <constraint exp="" field="pier_count" desc=""/>
    <constraint exp="" field="pier_width" desc=""/>
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
