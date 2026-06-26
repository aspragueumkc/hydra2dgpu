<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="bc_type">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="1" name="Wall (zero normal flux)" type="int"/>
            <Option value="2" name="Inflow Q (total discharge)" type="int"/>
            <Option value="3" name="Stage (prescribed WSE)" type="int"/>
            <Option value="6" name="Normal Depth (prescribed depth)" type="int"/>
            <Option value="7" name="Normal Depth (friction slope Sf)" type="int"/>
            <Option value="102" name="Timeseries Flow Q" type="int"/>
            <Option value="103" name="Timeseries Stage" type="int"/>
            <Option value="4" name="Open (zero-gradient)" type="int"/>
            <Option value="5" name="Reflecting" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="bc_value">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="priority">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="100" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="hydrograph">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="hydrograph_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="hydrograph_layer">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="BC Type"/>
    <alias index="1" name="BC Value"/>
    <alias index="2" name="Priority"/>
    <alias index="3" name="Hydrograph"/>
    <alias index="4" name="Hydrograph ID"/>
    <alias index="5" name="Hydrograph Layer"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp=""bc_type" IN (1,2,3,4,5,6,7,102,103)" field="bc_type" desc=""/>
      <constraint exp=""priority" &gt;= 0" field="priority" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="bc_type"/>
    <field name="bc_value"/>
    <field name="priority"/>
    <field name="hydrograph"/>
    <field name="hydrograph_id"/>
    <field name="hydrograph_layer"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>