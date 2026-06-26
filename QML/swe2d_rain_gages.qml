<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="gage_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
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
    <field name="hyetograph_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="units">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="mm/hr" name="mm/hr" type="string"/>
            <Option value="in/hr" name="in/hr" type="string"/>
            <Option value="mm" name="mm" type="string"/>
            <Option value="in" name="in" type="string"/>
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
    <alias index="0" name="Gage ID"/>
    <alias index="1" name="Name"/>
    <alias index="2" name="Hyetograph ID"/>
    <alias index="3" name="Units"/>
    <alias index="4" name="Priority"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp="length(trim(&quot;gage_id&quot;)) &gt; 0" field="gage_id" desc=""/>
      <constraint exp="length(trim(&quot;hyetograph_id&quot;)) &gt; 0" field="hyetograph_id" desc=""/>
      <constraint exp="&quot;units&quot; IS NULL OR &quot;units&quot; IN ('mm/hr','in/hr','mm','in')" field="units" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="gage_id"/>
    <field name="name"/>
    <field name="hyetograph_id"/>
    <field name="units"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>