<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="line_id">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="name">
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
    <field name="priority">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="100" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Line ID"/>
    <alias index="1" name="Name"/>
    <alias index="2" name="Enabled"/>
    <alias index="3" name="Priority"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp=""line_id" IS NULL OR "line_id" &gt;= 0" field="line_id" desc=""/>
      <constraint exp=""enabled" IS NULL OR "enabled" IN (0,1)" field="enabled" desc=""/>
      <constraint exp=""priority" IS NULL OR "priority" &gt;= 0" field="priority" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="line_id"/>
    <field name="name"/>
    <field name="enabled"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>