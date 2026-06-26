<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="arc_id">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="2147483647" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="node0">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="2147483647" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="node1">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="2147483647" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="use_global_arc_ctrl">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="1" name="Use global control" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="0" name="Per-arc override" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="arc_mode_override">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="hard_embed" name="Hard embed arcs" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="soft_size_hint" name="Soft arc size hint" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="disabled" name="Disable arc influence" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="arc_soft_size_override">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="arc_soft_dist_override">
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
  </aliases>
  <defaults>
    <default expression="" field="arc_id" applyOnUpdate="0"/>
    <default expression="" field="node0" applyOnUpdate="0"/>
    <default expression="" field="node1" applyOnUpdate="0"/>
    <default expression="" field="use_global_arc_ctrl" applyOnUpdate="0"/>
    <default expression="" field="arc_mode_override" applyOnUpdate="0"/>
    <default expression="" field="arc_soft_size_override" applyOnUpdate="0"/>
    <default expression="" field="arc_soft_dist_override" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="arc_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="node0" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="node1" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="use_global_arc_ctrl" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="arc_mode_override" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="arc_soft_size_override" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="arc_soft_dist_override" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="" field="arc_id" desc=""/>
    <constraint exp="" field="node0" desc=""/>
    <constraint exp="" field="node1" desc=""/>
    <constraint exp="&quot;use_global_arc_ctrl&quot; IS NULL OR &quot;use_global_arc_ctrl&quot; IN (0,1)" field="use_global_arc_ctrl" desc=""/>
    <constraint exp="&quot;arc_mode_override&quot; IS NULL OR &quot;arc_mode_override&quot; IN ('hard_embed','soft_size_hint','disabled')" field="arc_mode_override" desc=""/>
    <constraint exp="&quot;arc_soft_size_override&quot; IS NULL OR &quot;arc_soft_size_override&quot; &gt; 0" field="arc_soft_size_override" desc=""/>
    <constraint exp="&quot;arc_soft_dist_override&quot; IS NULL OR &quot;arc_soft_dist_override&quot; &gt; 0" field="arc_soft_dist_override" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="arc_id"/>
    <field name="node0"/>
    <field name="node1"/>
    <field name="use_global_arc_ctrl"/>
    <field name="arc_mode_override"/>
    <field name="arc_soft_size_override"/>
    <field name="arc_soft_dist_override"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
