<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="inlet_type_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="name">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="weir_length">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="orifice_area">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="coeff_weir">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="coeff_orifice">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="max_capture">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="description">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" field="inlet_type_id" name="Inlet Type ID"/>
    <alias index="1" field="name" name="Name"/>
    <alias index="2" field="weir_length" name="Weir Length"/>
    <alias index="3" field="orifice_area" name="Orifice Area"/>
    <alias index="4" field="coeff_weir" name="Weir Coefficient"/>
    <alias index="5" field="coeff_orifice" name="Orifice Coefficient"/>
    <alias index="6" field="max_capture" name="Max Capture"/>
    <alias index="7" field="description" name="Description"/>
  </aliases>
  <defaults>
    <default expression="" field="inlet_type_id" applyOnUpdate="0"/>
    <default expression="" field="name" applyOnUpdate="0"/>
    <default expression="" field="weir_length" applyOnUpdate="0"/>
    <default expression="" field="orifice_area" applyOnUpdate="0"/>
    <default expression="" field="coeff_weir" applyOnUpdate="0"/>
    <default expression="" field="coeff_orifice" applyOnUpdate="0"/>
    <default expression="" field="max_capture" applyOnUpdate="0"/>
    <default expression="" field="description" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="inlet_type_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="name" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="weir_length" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="orifice_area" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="coeff_weir" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="coeff_orifice" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="max_capture" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="description" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;inlet_type_id&quot;)) &gt; 0" field="inlet_type_id" desc=""/>
    <constraint exp="" field="name" desc=""/>
    <constraint exp="&quot;weir_length&quot; IS NULL OR &quot;weir_length&quot; &gt; 0" field="weir_length" desc=""/>
    <constraint exp="&quot;orifice_area&quot; IS NULL OR &quot;orifice_area&quot; &gt; 0" field="orifice_area" desc=""/>
    <constraint exp="&quot;coeff_weir&quot; IS NULL OR &quot;coeff_weir&quot; &gt; 0" field="coeff_weir" desc=""/>
    <constraint exp="&quot;coeff_orifice&quot; IS NULL OR &quot;coeff_orifice&quot; &gt; 0" field="coeff_orifice" desc=""/>
    <constraint exp="&quot;max_capture&quot; IS NULL OR &quot;max_capture&quot; &gt; 0" field="max_capture" desc=""/>
    <constraint exp="" field="description" desc=""/>
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
