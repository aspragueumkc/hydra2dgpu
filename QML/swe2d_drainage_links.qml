<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="link_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="from_node">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="to_node">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="link_type">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="conduit" name="Conduit" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="lateral_simple" name="Short lateral (simplified)" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="pump" name="Pump" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="weir" name="Weir" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="orifice" name="Orifice" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="culvert" name="Culvert (HDS-5)" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="link_shape">
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
                <Option value="pipe_arch" name="Pipe arch" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="custom" name="Custom area" type="string"/>
              </Option>
              </Option>
            </Option>
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
    <field configurationFlags="NoFlag" name="diameter">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="span">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="rise">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="area_m2">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="equiv_diameter_m">
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
    <field configurationFlags="NoFlag" name="cd">
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
  </fieldConfiguration>
  <aliases>
    <alias index="0" field="link_id" name="Link ID"/>
    <alias index="1" field="from_node" name="From Node"/>
    <alias index="2" field="to_node" name="To Node"/>
    <alias index="3" field="link_type" name="Link Type"/>
    <alias index="4" field="link_shape" name="Link Shape"/>
    <alias index="5" field="length" name="Length"/>
    <alias index="6" field="roughness_n" name="Roughness n"/>
    <alias index="7" field="diameter" name="Diameter"/>
    <alias index="8" field="span" name="Span"/>
    <alias index="9" field="rise" name="Rise"/>
    <alias index="10" field="area_m2" name="Area (m²)"/>
    <alias index="11" field="equiv_diameter_m" name="Equivalent Diameter"/>
    <alias index="12" field="max_flow" name="Max Flow"/>
    <alias index="13" field="cd" name="Discharge Coefficient"/>
    <alias index="14" field="culvert_shape" name="Culvert Shape"/>
    <alias index="15" field="culvert_code" name="Culvert Code"/>
    <alias index="16" field="culvert_rise" name="Culvert Rise"/>
    <alias index="17" field="culvert_span" name="Culvert Span"/>
    <alias index="18" field="culvert_area_m2" name="Culvert Area"/>
    <alias index="19" field="culvert_barrels" name="Culvert Barrels"/>
    <alias index="20" field="culvert_slope" name="Culvert Slope"/>
    <alias index="21" field="inlet_invert_elev" name="Inlet Invert Elev."/>
    <alias index="22" field="outlet_invert_elev" name="Outlet Invert Elev."/>
    <alias index="23" field="entrance_loss_k" name="Entrance Loss K"/>
    <alias index="24" field="exit_loss_k" name="Exit Loss K"/>
    <alias index="25" field="inlet_loss_k" name="Inlet Loss K"/>
    <alias index="26" field="outlet_loss_k" name="Outlet Loss K"/>
  </aliases>
  <defaults>
    <default expression="" field="link_id" applyOnUpdate="0"/>
    <default expression="" field="from_node" applyOnUpdate="0"/>
    <default expression="" field="to_node" applyOnUpdate="0"/>
    <default expression="" field="link_type" applyOnUpdate="0"/>
    <default expression="" field="link_shape" applyOnUpdate="0"/>
    <default expression="" field="length" applyOnUpdate="0"/>
    <default expression="" field="roughness_n" applyOnUpdate="0"/>
    <default expression="" field="diameter" applyOnUpdate="0"/>
    <default expression="" field="span" applyOnUpdate="0"/>
    <default expression="" field="rise" applyOnUpdate="0"/>
    <default expression="" field="area_m2" applyOnUpdate="0"/>
    <default expression="" field="equiv_diameter_m" applyOnUpdate="0"/>
    <default expression="" field="max_flow" applyOnUpdate="0"/>
    <default expression="" field="cd" applyOnUpdate="0"/>
    <default expression="" field="culvert_shape" applyOnUpdate="0"/>
    <default expression="" field="culvert_code" applyOnUpdate="0"/>
    <default expression="" field="culvert_rise" applyOnUpdate="0"/>
    <default expression="" field="culvert_span" applyOnUpdate="0"/>
    <default expression="" field="culvert_area_m2" applyOnUpdate="0"/>
    <default expression="" field="culvert_barrels" applyOnUpdate="0"/>
    <default expression="" field="culvert_slope" applyOnUpdate="0"/>
    <default expression="" field="inlet_invert_elev" applyOnUpdate="0"/>
    <default expression="" field="outlet_invert_elev" applyOnUpdate="0"/>
    <default expression="" field="entrance_loss_k" applyOnUpdate="0"/>
    <default expression="" field="exit_loss_k" applyOnUpdate="0"/>
    <default expression="" field="inlet_loss_k" applyOnUpdate="0"/>
    <default expression="" field="outlet_loss_k" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="link_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="from_node" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="to_node" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="link_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="link_shape" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="length" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="roughness_n" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="diameter" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="span" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="rise" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="area_m2" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="equiv_diameter_m" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="max_flow" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="cd" unique_strength="0"/>
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
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="inlet_loss_k" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="outlet_loss_k" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;link_id&quot;)) &gt; 0" field="link_id" desc=""/>
    <constraint exp="length(trim(&quot;from_node&quot;)) &gt; 0" field="from_node" desc=""/>
    <constraint exp="length(trim(&quot;to_node&quot;)) &gt; 0" field="to_node" desc=""/>
    <constraint exp="&quot;link_type&quot; IN ('conduit','lateral_simple','pump','weir','orifice','culvert')" field="link_type" desc=""/>
    <constraint exp="&quot;link_shape&quot; IS NULL OR &quot;link_shape&quot; IN ('circular','box','pipe_arch','custom')" field="link_shape" desc=""/>
    <constraint exp="&quot;length&quot; IS NULL OR &quot;length&quot; &gt; 0" field="length" desc=""/>
    <constraint exp="&quot;roughness_n&quot; IS NULL OR &quot;roughness_n&quot; &gt; 0" field="roughness_n" desc=""/>
    <constraint exp="&quot;diameter&quot; IS NULL OR &quot;diameter&quot; &gt; 0" field="diameter" desc=""/>
    <constraint exp="&quot;span&quot; IS NULL OR &quot;span&quot; &gt; 0" field="span" desc=""/>
    <constraint exp="&quot;rise&quot; IS NULL OR &quot;rise&quot; &gt; 0" field="rise" desc=""/>
    <constraint exp="&quot;area_m2&quot; IS NULL OR &quot;area_m2&quot; &gt; 0" field="area_m2" desc=""/>
    <constraint exp="" field="equiv_diameter_m" desc=""/>
    <constraint exp="" field="max_flow" desc=""/>
    <constraint exp="" field="cd" desc=""/>
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
    <constraint exp="" field="inlet_loss_k" desc=""/>
    <constraint exp="" field="outlet_loss_k" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="link_id"/>
    <field name="from_node"/>
    <field name="to_node"/>
    <field name="link_type"/>
    <field name="link_shape"/>
    <field name="length"/>
    <field name="roughness_n"/>
    <field name="diameter"/>
    <field name="span"/>
    <field name="rise"/>
    <field name="area_m2"/>
    <field name="equiv_diameter_m"/>
    <field name="max_flow"/>
    <field name="cd"/>
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
    <field name="inlet_loss_k"/>
    <field name="outlet_loss_k"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
