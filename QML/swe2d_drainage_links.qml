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
