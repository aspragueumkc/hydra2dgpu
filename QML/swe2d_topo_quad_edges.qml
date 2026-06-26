<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="region_id">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="edge_id">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="1" name="Edge 1 (left)" type="int"/>
            <Option value="2" name="Edge 2 (right)" type="int"/>
            <Option value="3" name="Edge 3 (bottom)" type="int"/>
            <Option value="4" name="Edge 4 (top)" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="target_size">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="n_layers">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="1000" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="first_height">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="growth_rate">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Region ID"/>
    <alias index="1" name="Edge ID"/>
    <alias index="2" name="Target Size"/>
    <alias index="3" name="Number of Layers"/>
    <alias index="4" name="First Cell Height"/>
    <alias index="5" name="Growth Rate"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp=""region_id" &gt;= 0" field="region_id" desc=""/>
      <constraint exp=""edge_id" IN (1,2,3,4)" field="edge_id" desc=""/>
      <constraint exp=""target_size" IS NULL OR "target_size" &gt; 0" field="target_size" desc=""/>
      <constraint exp=""n_layers" &gt;= 0" field="n_layers" desc=""/>
      <constraint exp=""first_height" IS NULL OR "first_height" &gt; 0" field="first_height" desc=""/>
      <constraint exp=""growth_rate" IS NULL OR "growth_rate" &gt; 0" field="growth_rate" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="region_id"/>
    <field name="edge_id"/>
    <field name="target_size"/>
    <field name="n_layers"/>
    <field name="first_height"/>
    <field name="growth_rate"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>