<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="zone_id">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="cn">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="100" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
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
    <alias index="0" name="Zone ID"/>
    <alias index="1" name="Curve Number"/>
    <alias index="2" name="Priority"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp=""cn" &gt;= 1 AND "cn" &lt;= 100" field="cn" desc=""/>
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
    <field name="zone_id"/>
    <field name="cn"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>