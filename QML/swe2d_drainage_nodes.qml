<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="node_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="invert_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="max_depth">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="rim_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="crest_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="node_type">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="junction" name="Junction" type="string"/>
            <Option value="outfall" name="Outfall" type="string"/>
            <Option value="storage" name="Storage" type="string"/>
            <Option value="inlet" name="Inlet" type="string"/>
            <Option value="pipe_end" name="Pipe end" type="string"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="surface_area">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="outfall_area">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="zero_storage">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="0" name="No" type="int"/>
            <Option value="1" name="Yes" type="int"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Node ID"/>
    <alias index="1" name="Invert Elevation"/>
    <alias index="2" name="Max Depth"/>
    <alias index="3" name="Rim Elevation"/>
    <alias index="4" name="Crest Elevation"/>
    <alias index="5" name="Node Type"/>
    <alias index="6" name="Surface Area"/>
    <alias index="7" name="Outfall Area"/>
    <alias index="8" name="Zero Storage"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp="length(trim(&quot;node_id&quot;)) &gt; 0" field="node_id" desc=""/>
      <constraint exp="&quot;node_type&quot; IN ('junction','outfall','storage','inlet','pipe_end')" field="node_type" desc=""/>
      <constraint exp="&quot;max_depth&quot; IS NULL OR &quot;max_depth&quot; &gt; 0" field="max_depth" desc=""/>
      <constraint exp="&quot;rim_elev&quot; IS NULL OR &quot;rim_elev&quot; &gt;= &quot;invert_elev&quot;" field="rim_elev" desc=""/>
      <constraint exp="&quot;crest_elev&quot; IS NULL OR &quot;crest_elev&quot; &gt;= &quot;invert_elev&quot;" field="crest_elev" desc=""/>
      <constraint exp="&quot;surface_area&quot; IS NULL OR &quot;surface_area&quot; &gt; 0" field="surface_area" desc=""/>
      <constraint exp="&quot;outfall_area&quot; IS NULL OR &quot;outfall_area&quot; &gt; 0" field="outfall_area" desc=""/>
      <constraint exp="&quot;zero_storage&quot; IS NULL OR &quot;zero_storage&quot; IN (0,1)" field="zero_storage" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="node_id"/>
    <field name="invert_elev"/>
    <field name="max_depth"/>
    <field name="rim_elev"/>
    <field name="crest_elev"/>
    <field name="node_type"/>
    <field name="surface_area"/>
    <field name="outfall_area"/>
    <field name="zero_storage"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>