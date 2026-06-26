<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="constraint_id">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="2147483647" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
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
    <field name="cell_type">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="triangular" name="Triangular" type="string"/>
            <Option value="quadrilateral" name="Quadrilateral" type="string"/>
            <Option value="cartesian" name="Cartesian" type="string"/>
            <Option value="channel_generator" name="Channel generator" type="string"/>
            <Option value="empty" name="Empty" type="string"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="edge_len_1">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="edge_len_2">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="edge_len_3">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="edge_len_4">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Constraint ID"/>
    <alias index="1" name="Target Size"/>
    <alias index="2" name="Cell Type"/>
    <alias index="3" name="Edge Length 1"/>
    <alias index="4" name="Edge Length 2"/>
    <alias index="5" name="Edge Length 3"/>
    <alias index="6" name="Edge Length 4"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp=""cell_type" IN ('triangular','quadrilateral','cartesian','channel_generator','empty')" field="cell_type" desc=""/>
      <constraint exp=""target_size" &gt; 0" field="target_size" desc=""/>
      <constraint exp=""edge_len_1" IS NULL OR "edge_len_1" &gt; 0" field="edge_len_1" desc=""/>
      <constraint exp=""edge_len_2" IS NULL OR "edge_len_2" &gt; 0" field="edge_len_2" desc=""/>
      <constraint exp=""edge_len_3" IS NULL OR "edge_len_3" &gt; 0" field="edge_len_3" desc=""/>
      <constraint exp=""edge_len_4" IS NULL OR "edge_len_4" &gt; 0" field="edge_len_4" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="constraint_id"/>
    <field name="target_size"/>
    <field name="cell_type"/>
    <field name="edge_len_1"/>
    <field name="edge_len_2"/>
    <field name="edge_len_3"/>
    <field name="edge_len_4"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>