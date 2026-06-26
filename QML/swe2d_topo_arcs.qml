<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="arc_id">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="node0">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="node1">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="use_global_arc_ctrl">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="1" name="Use global control" type="int"/>
            <Option value="0" name="Per-arc override" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="arc_mode_override">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="hard_embed" name="Hard embed arcs" type="string"/>
            <Option value="soft_size_hint" name="Soft arc size hint" type="string"/>
            <Option value="disabled" name="Disable arc influence" type="string"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="arc_soft_size_override">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="arc_soft_dist_override">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Arc ID"/>
    <alias index="1" name="From Node"/>
    <alias index="2" name="To Node"/>
    <alias index="3" name="Use Global Arc Ctrl"/>
    <alias index="4" name="Arc Mode Override"/>
    <alias index="5" name="Arc Soft Size Override"/>
    <alias index="6" name="Arc Soft Dist Override"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp=""use_global_arc_ctrl" IS NULL OR "use_global_arc_ctrl" IN (0,1)" field="use_global_arc_ctrl" desc=""/>
      <constraint exp=""arc_mode_override" IS NULL OR "arc_mode_override" IN ('hard_embed','soft_size_hint','disabled')" field="arc_mode_override" desc=""/>
      <constraint exp=""arc_soft_size_override" IS NULL OR "arc_soft_size_override" &gt; 0" field="arc_soft_size_override" desc=""/>
      <constraint exp=""arc_soft_dist_override" IS NULL OR "arc_soft_dist_override" &gt; 0" field="arc_soft_dist_override" desc=""/>
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