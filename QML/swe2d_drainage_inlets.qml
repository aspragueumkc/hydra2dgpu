<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="inlet_type_id">
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
    <field name="weir_length">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="orifice_area">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="coeff_weir">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="coeff_orifice">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="max_capture">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
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
    <alias index="0" name="Inlet Type ID"/>
    <alias index="1" name="Name"/>
    <alias index="2" name="Weir Length"/>
    <alias index="3" name="Orifice Area"/>
    <alias index="4" name="Weir Coefficient"/>
    <alias index="5" name="Orifice Coefficient"/>
    <alias index="6" name="Max Capture"/>
    <alias index="7" name="Description"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp="length(trim("inlet_type_id")) &gt; 0" field="inlet_type_id" desc=""/>
      <constraint exp=""weir_length" IS NULL OR "weir_length" &gt; 0" field="weir_length" desc=""/>
      <constraint exp=""orifice_area" IS NULL OR "orifice_area" &gt; 0" field="orifice_area" desc=""/>
      <constraint exp=""coeff_weir" IS NULL OR "coeff_weir" &gt; 0" field="coeff_weir" desc=""/>
      <constraint exp=""coeff_orifice" IS NULL OR "coeff_orifice" &gt; 0" field="coeff_orifice" desc=""/>
      <constraint exp=""max_capture" IS NULL OR "max_capture" &gt; 0" field="max_capture" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="inlet_type_id"/>
    <field name="name"/>
    <field name="weir_length"/>
    <field name="orifice_area"/>
    <field name="coeff_weir"/>
    <field name="coeff_orifice"/>
    <field name="max_capture"/>
    <field name="description"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>