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
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="circular" name="Circular" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="box" name="Box" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="rectangular" name="Rectangular" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="culvert_code">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="0" name="— Select culvert code —" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="1" name="Circular concrete, square edge w/ headwall" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="2" name="Circular concrete, groove end w/ headwall" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="3" name="Circular concrete, groove end projecting" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="4" name="Circular concrete, mitred to slope" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="5" name="Circular concrete, beveled ring" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="6" name="Circular concrete, beveled ring (smoother)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="7" name="Circular CMP, projecting" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="8" name="Circular CMP, projecting (different edge)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="9" name="Circular CMP, mitered to slope" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="10" name="Circular CMP, mitered to slope (alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="11" name="Circular CMP, beveled end (thin wall)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="12" name="Circular CMP, groove end in headwall" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="13" name="Circular CMP, groove end in headwall (alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="14" name="Circular CMP, headwall (square edge)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="15" name="Circular CMP, headwall (groove end)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="16" name="Circular CMP, headwall (thin wall projecting)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="17" name="Rectangular box, 30-75deg wingwall flares" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="18" name="Rectangular box, 90deg headwall w/ chamfers" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="19" name="Rectangular box, 0deg wingwall flares" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="20" name="Rectangular box, 45deg wingwall flares" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="21" name="Rectangular box, 18-33deg wingwall flares" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="22" name="Rectangular box, 0deg wingwall flares (thick)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="23" name="Rectangular box, 30deg wingwall flares (thick)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="24" name="Rectangular box, 45deg wingwall flares (thick)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="25" name="Rectangular box, 0deg wingwall flares (thick alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="26" name="Rectangular box, beveled edge (1:1)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="27" name="Circular concrete, square edge w/ headwall (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="28" name="Circular concrete, groove end w/ headwall (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="29" name="Circular concrete, groove end projecting (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="30" name="Circular CMP, projecting (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="31" name="Circular CMP, mitered to slope (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="32" name="Circular CMP, beveled end thin wall (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="33" name="Circular CMP, groove end in headwall (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="34" name="Circular CMP, headwall square edge (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="35" name="Circular CMP, headwall groove end (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="36" name="Circular CMP, beveled ring (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="37" name="Circular CMP, beveled ring thick (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="38" name="Circular concrete, beveled ring (form-1 alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="39" name="Circular pipe, beveled ring (thin wall)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="40" name="Circular pipe, beveled ring (thick wall)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="41" name="Circular pipe, 45deg beveled ring" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="42" name="Circular pipe, 33.7deg beveled ring" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="43" name="Circular pipe, 45deg bevel (offset)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="44" name="Circular pipe, 33.7deg bevel (offset)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="45" name="Circular CMP, prefab end section (safety)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="46" name="Circular CMP, prefab end section (alt)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="47" name="Arch CMP, 2-3-1 fill (soffit thickness 0.0625)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="48" name="Arch CMP, 2-3-1 fill (soffit varying)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="49" name="Arch CMP, 2-3-1 fill projecting (soffit varying)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="50" name="Arch CMP, 2-2-1 fill (soffit thickness 0.0625)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="51" name="Pipe arch CMP, 0.75x0.75 fill (soffit thickness 0.0625)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="52" name="Pipe arch CMP, 0.75x0.75 fill projecting" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="53" name="Pipe arch CMP, 0.75x0.75 fill (soffit varying)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="54" name="Horizontal ellipse, concrete (form-2)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="55" name="Horizontal ellipse, corrugated metal (form-2)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="56" name="Arch CMP, 2-3-1 fill premium (form-2)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="57" name="Horizontal ellipse, special shape (form-2)" type="int"/>
              </Option>
              </Option>
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
    <alias index="0" field="structure_id" name="Structure ID"/>
    <alias index="1" field="structure_type" name="Structure Type"/>
    <alias index="2" field="crest_elev" name="Crest Elevation"/>
    <alias index="3" field="enabled" name="Enabled"/>
    <alias index="4" field="width" name="Width"/>
    <alias index="5" field="height" name="Height"/>
    <alias index="6" field="diameter" name="Diameter"/>
    <alias index="7" field="culvert_shape" name="Culvert Shape"/>
    <alias index="8" field="culvert_code" name="Culvert Code"/>
    <alias index="9" field="culvert_rise" name="Culvert Rise"/>
    <alias index="10" field="culvert_span" name="Culvert Span"/>
    <alias index="11" field="culvert_area_m2" name="Culvert Area"/>
    <alias index="12" field="culvert_barrels" name="Barrels"/>
    <alias index="13" field="culvert_slope" name="Culvert Slope"/>
    <alias index="14" field="inlet_invert_elev" name="Inlet Invert Elev."/>
    <alias index="15" field="outlet_invert_elev" name="Outlet Invert Elev."/>
    <alias index="16" field="entrance_loss_k" name="Entrance Loss K"/>
    <alias index="17" field="exit_loss_k" name="Exit Loss K"/>
    <alias index="18" field="embankment_enabled" name="Embankment Enabled"/>
    <alias index="19" field="embankment_crest_elev" name="Embankment Crest Elev."/>
    <alias index="20" field="embankment_overflow_width" name="Overflow Width"/>
    <alias index="21" field="embankment_weir_coeff" name="Weir Coefficient"/>
    <alias index="22" field="length" name="Length"/>
    <alias index="23" field="roughness_n" name="Roughness n"/>
    <alias index="24" field="coeff" name="Coefficient"/>
    <alias index="25" field="cd" name="Discharge Coeff."/>
    <alias index="26" field="opening" name="Opening"/>
    <alias index="27" field="q_pump" name="Pump Flow"/>
    <alias index="28" field="max_flow" name="Max Flow"/>
    <alias index="29" field="inlet_loss_k" name="Inlet Loss K"/>
    <alias index="30" field="outlet_loss_k" name="Outlet Loss K"/>
    <alias index="31" field="stacked_enabled" name="Stacked Enabled"/>
    <alias index="32" field="use_redistribution" name="Use Redistribution"/>
    <alias index="33" field="influence_width" name="Influence Width"/>
    <alias index="34" field="upstream_buffer" name="Upstream Buffer"/>
    <alias index="35" field="downstream_buffer" name="Downstream Buffer"/>
    <alias index="36" field="deck_soffit_elev" name="Deck Soffit Elev."/>
    <alias index="37" field="deck_top_elev" name="Deck Top Elev."/>
    <alias index="38" field="model_top_elev" name="Model Top Elev."/>
    <alias index="39" field="under_layers" name="Under Layers"/>
    <alias index="40" field="over_layers" name="Over Layers"/>
    <alias index="41" field="pier_count" name="Pier Count"/>
    <alias index="42" field="pier_width" name="Pier Width"/>
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
  <attributeEditorForm>
    <attributeEditorContainer name="Weir" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 1" groupBox="0">
      <attributeEditorField index="-1" name="width"/>
      <attributeEditorField index="-1" name="embankment_enabled"/>
      <attributeEditorField index="-1" name="embankment_crest_elev"/>
      <attributeEditorField index="-1" name="embankment_overflow_width"/>
      <attributeEditorField index="-1" name="embankment_weir_coeff"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Culvert" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 2" groupBox="0">
      <attributeEditorField index="-1" name="culvert_shape"/>
      <attributeEditorField index="-1" name="culvert_code"/>
      <attributeEditorField index="-1" name="culvert_rise"/>
      <attributeEditorField index="-1" name="culvert_span"/>
      <attributeEditorField index="-1" name="culvert_area_m2"/>
      <attributeEditorField index="-1" name="culvert_barrels"/>
      <attributeEditorField index="-1" name="culvert_slope"/>
      <attributeEditorField index="-1" name="diameter"/>
      <attributeEditorField index="-1" name="length"/>
      <attributeEditorField index="-1" name="roughness_n"/>
      <attributeEditorField index="-1" name="inlet_invert_elev"/>
      <attributeEditorField index="-1" name="outlet_invert_elev"/>
      <attributeEditorField index="-1" name="entrance_loss_k"/>
      <attributeEditorField index="-1" name="exit_loss_k"/>
      <attributeEditorField index="-1" name="embankment_enabled"/>
      <attributeEditorField index="-1" name="embankment_crest_elev"/>
      <attributeEditorField index="-1" name="embankment_overflow_width"/>
      <attributeEditorField index="-1" name="embankment_weir_coeff"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Gate" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 3" groupBox="0">
      <attributeEditorField index="-1" name="width"/>
      <attributeEditorField index="-1" name="height"/>
      <attributeEditorField index="-1" name="opening"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Bridge" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 4" groupBox="0">
      <attributeEditorField index="-1" name="width"/>
      <attributeEditorField index="-1" name="length"/>
      <attributeEditorField index="-1" name="deck_soffit_elev"/>
      <attributeEditorField index="-1" name="deck_top_elev"/>
      <attributeEditorField index="-1" name="model_top_elev"/>
      <attributeEditorField index="-1" name="under_layers"/>
      <attributeEditorField index="-1" name="over_layers"/>
      <attributeEditorField index="-1" name="inlet_loss_k"/>
      <attributeEditorField index="-1" name="outlet_loss_k"/>
      <attributeEditorField index="-1" name="pier_count"/>
      <attributeEditorField index="-1" name="pier_width"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Pump" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 5" groupBox="0">
      <attributeEditorField index="-1" name="q_pump"/>
      <attributeEditorField index="-1" name="max_flow"/>
    </attributeEditorContainer>
  </attributeEditorForm>
</qgis>
