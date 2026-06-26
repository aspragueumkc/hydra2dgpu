<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="hyetograph_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="Time">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="Value">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="value_type">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="intensity" name="Intensity" type="string"/>
            <Option value="incremental" name="Incremental depth" type="string"/>
            <Option value="cumulative" name="Cumulative depth" type="string"/>
            </Option>
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
    <field name="description">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Hyetograph ID"/>
    <alias index="1" name="Time"/>
    <alias index="2" name="Value"/>
    <alias index="3" name="Value Type"/>
    <alias index="4" name="Units"/>
    <alias index="5" name="Description"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp="length(trim("hyetograph_id")) &gt; 0" field="hyetograph_id" desc=""/>
      <constraint exp="length(trim("Time")) &gt; 0" field="Time" desc=""/>
      <constraint exp=""Value" &gt;= 0" field="Value" desc=""/>
      <constraint exp=""value_type" IS NULL OR "value_type" IN ('intensity','incremental','cumulative')" field="value_type" desc=""/>
      <constraint exp=""units" IS NULL OR "units" IN ('mm/hr','in/hr','mm','in')" field="units" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="hyetograph_id"/>
    <field name="Time"/>
    <field name="Value"/>
    <field name="value_type"/>
    <field name="units"/>
    <field name="description"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>