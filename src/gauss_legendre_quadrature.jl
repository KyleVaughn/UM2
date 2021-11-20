# Weights and points for Gauss-Legendre quadrature for a variety of shapes

function gauss_legendre_quadrature(type::Type{F}, N::Int64) where {F <: AbstractFloat}
    # Fhe weights and points for Gauss-Legendre quadrature on the interval [0,1]
    # ∑wᵢ= 1, rᵢ∈ [0, 1]
    #
    # N that have entries in this function: N = [ 1, 2, 3, 4, 5, 10, 15, 20].
    if N === 1
        weights = F[1]
        r = F[0.5]
    elseif N === 2
        weights = F[0.5, 0.5]
        r = F[0.21132486540518713, 0.7886751345948129]
    elseif N === 3
        weights = F[0.2777777777777776,
                    0.4444444444444444,
                    0.2777777777777776]
        r = F[0.1127016653792583,
              0.5,
              0.8872983346207417]
    elseif N === 4
        weights = F[
                      0.17392742256872684,
                      0.3260725774312732,
                      0.3260725774312732,
                      0.17392742256872684
                   ]
        r = F[
              0.06943184420297371,
              0.33000947820757187,
              0.6699905217924281,
              0.9305681557970262
             ]
    elseif N === 5
        weights = F[
                      0.1184634425280945,
                      0.23931433524968326,
                      0.28444444444444444,
                      0.23931433524968326,
                      0.1184634425280945
                   ]
        r = F[
              0.04691007703066802,
              0.23076534494715845,
              0.5,
              0.7692346550528415,
              0.9530899229693319,
             ]
    elseif N === 10
        weights = F[
                      0.03333567215434387,
                      0.07472567457529025,
                      0.1095431812579911,
                      0.1346333596549981,
                      0.14776211235737646,
                      0.14776211235737646,
                      0.1346333596549981,
                      0.1095431812579911,
                      0.07472567457529025,
                      0.03333567215434387
                   ]
        r = F[
              0.013046735741414184,
              0.06746831665550773,
              0.16029521585048778,
              0.2833023029353764,
              0.4255628305091844,
              0.5744371694908156,
              0.7166976970646236,
              0.8397047841495122,
              0.9325316833444923,
              0.9869532642585859
             ]
    elseif N === 15
        weights = F[
                        0.015376620998058315,
                        0.03518302374405407,
                        0.053579610233586,
                        0.06978533896307715,
                        0.083134602908497,
                        0.0930805000077811,
                        0.0992157426635558,
                        0.10128912096278064,
                        0.0992157426635558,
                        0.0930805000077811,
                        0.083134602908497,
                        0.06978533896307715,
                        0.053579610233586,
                        0.03518302374405407,
                        0.015376620998058315
                    ]
        r = F[
              0.006003740989757311,
              0.031363303799647024,
              0.0758967082947864,
              0.13779113431991497,
              0.21451391369573058,
              0.3029243264612183,
              0.39940295300128276,
              0.5,
              0.6005970469987172,
              0.6970756735387817,
              0.7854860863042694,
              0.862208865680085,
              0.9241032917052137,
              0.968636696200353,
              0.9939962590102427
             ]
    elseif N === 20
        weights = F[
                      0.00880700356957606,
                      0.02030071490019347,
                      0.03133602416705453,
                      0.04163837078835238,
                      0.0509650599086202,
                      0.0590972659807592,
                      0.0658443192245883,
                      0.07104805465919105,
                      0.07458649323630186,
                      0.07637669356536295,
                      0.07637669356536295,
                      0.07458649323630186,
                      0.07104805465919105,
                      0.0658443192245883,
                      0.0590972659807592,
                      0.0509650599086202,
                      0.04163837078835238,
                      0.03133602416705453,
                      0.02030071490019347,
                      0.00880700356957606
                   ]
        r = F[
              0.0034357004074525577,
              0.018014036361043095,
              0.04388278587433703,
              0.08044151408889061,
              0.1268340467699246,
              0.1819731596367425,
              0.24456649902458644,
              0.3131469556422902,
              0.38610707442917747,
              0.46173673943325133,
              0.5382632605667487,
              0.6138929255708225,
              0.6868530443577098,
              0.7554335009754136,
              0.8180268403632576,
              0.8731659532300754,
              0.9195584859111094,
              0.956117214125663,
              0.981985963638957,
              0.9965642995925474
             ]
    else
        weights = F[]
        r= F[]
    end
    return weights, r
end

function gauss_legendre_quadrature(tri::Triangle6_2D{F}, N::Int64) where {F <: AbstractFloat}
    # Fhe weights and points for Gauss-Legendre quadrature on the parametric unit triangle
    # ∑wᵢ= 1/2, rᵢ∈ [0, 1], sᵢ∈ [0, 1], rᵢ + sᵢ ≤ 1
    # N that have entries in this function: N = [12, 27, 48, 79]
    if N === 12
        # P6. 0 negative weights, 0 points outside of the triangle
        w = F[0.058393137863189,
              0.058393137863189,
              0.058393137863189,
              0.025422453185104,
              0.025422453185104,
              0.025422453185104,
              0.041425537809187,
              0.041425537809187,
              0.041425537809187,
              0.041425537809187,
              0.041425537809187,
              0.041425537809187]

        r = F[0.249286745170910,
              0.249286745170910,
              0.501426509658179,
              0.063089014491502,
              0.063089014491502,
              0.873821971016996,
              0.310352451033785,
              0.636502499121399,
              0.053145049844816,
              0.310352451033785,
              0.636502499121399,
              0.053145049844816]

        s = F[0.249286745170910,
              0.501426509658179,
              0.249286745170910,
              0.063089014491502,
              0.873821971016996,
              0.063089014491502,
              0.636502499121399,
              0.053145049844816,
              0.310352451033785,
              0.053145049844816,
              0.310352451033785,
              0.636502499121399]

    elseif N === 27
        # P11. 0 negative weights, 3 points outside of the triangle
        w = F[0.00046350316448,
              0.00046350316448,
              0.00046350316448,
              0.03857476745740,
              0.03857476745740,
              0.03857476745740,
              0.02966148869040,
              0.02966148869040,
              0.02966148869040,
              0.01809227025170,
              0.01809227025170,
              0.01809227025170,
              0.00682986550135,
              0.00682986550135,
              0.00682986550135,
              0.02616855598110,
              0.02616855598110,
              0.02616855598110,
              0.02616855598110,
              0.02616855598110,
              0.02616855598110,
              0.01035382981955,
              0.01035382981955,
              0.01035382981955,
              0.01035382981955,
              0.01035382981955,
              0.01035382981955]

        r = F[+0.5346110482710,
              -0.0692220965415,
              +0.5346110482710,
              +0.3989693029660,
              +0.2020613940680,
              +0.3989693029660,
              +0.2033099004310,
              +0.5933801991370,
              +0.2033099004310,
              +0.1193509122830,
              +0.7612981754350,
              +0.1193509122830,
              +0.0323649481113,
              +0.9352701037770,
              +0.0323649481113,
              +0.5932012134280,
              +0.0501781383105,
              +0.3566206482610,
              +0.0501781383105,
              +0.3566206482610,
              +0.5932012134280,
              +0.8074890031600,
              +0.0210220165362,
              +0.1714889803040,
              +0.0210220165362,
              +0.1714889803040,
              +0.8074890031600]
        
        s = F[+0.5346110482710,
              +0.5346110482710,
              -0.0692220965415,
              +0.3989693029660,
              +0.3989693029660,
              +0.2020613940680,
              +0.2033099004310,
              +0.2033099004310,
              +0.5933801991370,
              +0.1193509122830,
              +0.1193509122830,
              +0.7612981754350,
              +0.0323649481113,
              +0.0323649481113,
              +0.9352701037770,
              +0.3566206482610,
              +0.5932012134280,
              +0.0501781383105,
              +0.3566206482610,
              +0.5932012134280,
              +0.0501781383105,
              +0.1714889803040,
              +0.8074890031600,
              +0.0210220165362,
              +0.1714889803040,
              +0.8074890031600,
              +0.0210220165362]

    elseif N === 48
        # P15. 0 negative weights, 9 points outside of the triangle
        w = F[0.000958437821425,
              0.000958437821425,
              0.000958437821425,
              0.022124513635550,
              0.022124513635550,
              0.022124513635550,
              0.025593274359450,
              0.025593274359450,
              0.025593274359450,
              0.011843867935350,
              0.011843867935350,
              0.011843867935350,
              0.006644887845000,
              0.006644887845000,
              0.006644887845000,
              0.002374458304095,
              0.002374458304095,
              0.002374458304095,
              0.019275036299800,
              0.019275036299800,
              0.019275036299800,
              0.019275036299800,
              0.019275036299800,
              0.019275036299800,
              0.013607907160300,
              0.013607907160300,
              0.013607907160300,
              0.013607907160300,
              0.013607907160300,
              0.013607907160300,
              0.001091038683400,
              0.001091038683400,
              0.001091038683400,
              0.001091038683400,
              0.001091038683400,
              0.001091038683400,
              0.010752659923850,
              0.010752659923850,
              0.010752659923850,
              0.010752659923850,
              0.010752659923850,
              0.010752659923850,
              0.003836971315525,
              0.003836971315525,
              0.003836971315525,
              0.003836971315525,
              0.003836971315525,
              0.003836971315525]

        r = F[+0.5069729168580,
              -0.0139458337165,
              +0.5069729168580,
              +0.4314063542830,
              +0.1371872914340,
              +0.4314063542830,
              +0.2776936448470,
              +0.4446127103060,
              +0.2776936448470,
              +0.1264648910410,
              +0.7470702179170,
              +0.1264648910410,
              +0.0708083859747,
              +0.8583832280510,
              +0.0708083859747,
              +0.0189651702411,
              +0.9620696595180,
              +0.0189651702411,
              +0.6049544668930,
              +0.1337341619670,
              +0.2613113711400,
              +0.1337341619670,
              +0.2613113711400,
              +0.6049544668930,
              +0.5755865555130,
              +0.0363666773969,
              +0.3880467670900,
              +0.0363666773969,
              +0.3880467670900,
              +0.5755865555130,
              +0.7244626630770,
              -0.0101748831266,
              +0.2857122200500,
              -0.0101748831266,
              +0.2857122200500,
              +0.7244626630770,
              +0.7475564660520,
              +0.0368438698759,
              +0.2155996640720,
              +0.0368438698759,
              +0.2155996640720,
              +0.7475564660520,
              +0.8839645740920,
              +0.0124598093312,
              +0.1035756165760,
              +0.0124598093312,
              +0.1035756165760,
              +0.8839645740920]

        s = F[+0.5069729168580,
              +0.5069729168580,
              -0.0139458337165,
              +0.4314063542830,
              +0.4314063542830,
              +0.1371872914340,
              +0.2776936448470,
              +0.2776936448470,
              +0.4446127103060,
              +0.1264648910410,
              +0.1264648910410,
              +0.7470702179170,
              +0.0708083859747,
              +0.0708083859747,
              +0.8583832280510,
              +0.0189651702411,
              +0.0189651702411,
              +0.9620696595180,
              +0.2613113711400,
              +0.6049544668930,
              +0.1337341619670,
              +0.2613113711400,
              +0.6049544668930,
              +0.1337341619670,
              +0.3880467670900,
              +0.5755865555130,
              +0.0363666773969,
              +0.3880467670900,
              +0.5755865555130,
              +0.0363666773969,
              +0.2857122200500,
              +0.7244626630770,
              -0.0101748831266,
              +0.2857122200500,
              +0.7244626630770,
              -0.0101748831266,
              +0.2155996640720,
              +0.7475564660520,
              +0.0368438698759,
              +0.2155996640720,
              +0.7475564660520,
              +0.0368438698759,
              +0.1035756165760,
              +0.8839645740920,
              +0.0124598093312,
              +0.1035756165760,
              +0.8839645740920,
              +0.0124598093312]

    elseif N === 79
        # P20. 3 negative weights, 9 points outside of the triangle
        w = F[+0.016528527770800,
              +0.000433509592831,
              +0.000433509592831,
              +0.000433509592831,
              +0.005830026358200,
              +0.005830026358200,
              +0.005830026358200,
              +0.011438468178200,
              +0.011438468178200,
              +0.011438468178200,
              +0.015224491336950,
              +0.015224491336950,
              +0.015224491336950,
              +0.015312445862700,
              +0.015312445862700,
              +0.015312445862700,
              +0.012184028838400,
              +0.012184028838400,
              +0.012184028838400,
              +0.007998716016000,
              +0.007998716016000,
              +0.007998716016000,
              +0.003849150907800,
              +0.003849150907800,
              +0.003849150907800,
              -0.000316030248744,
              -0.000316030248744,
              -0.000316030248744,
              +0.000875567150595,
              +0.000875567150595,
              +0.000875567150595,
              +0.008232919594800,
              +0.008232919594800,
              +0.008232919594800,
              +0.008232919594800,
              +0.008232919594800,
              +0.008232919594800,
              +0.002419516770245,
              +0.002419516770245,
              +0.002419516770245,
              +0.002419516770245,
              +0.002419516770245,
              +0.002419516770245,
              +0.012902453267350,
              +0.012902453267350,
              +0.012902453267350,
              +0.012902453267350,
              +0.012902453267350,
              +0.012902453267350,
              +0.004235545527220,
              +0.004235545527220,
              +0.004235545527220,
              +0.004235545527220,
              +0.004235545527220,
              +0.004235545527220,
              +0.009177457053150,
              +0.009177457053150,
              +0.009177457053150,
              +0.009177457053150,
              +0.009177457053150,
              +0.009177457053150,
              +0.000352202338954,
              +0.000352202338954,
              +0.000352202338954,
              +0.000352202338954,
              +0.000352202338954,
              +0.000352202338954,
              +0.005056342463750,
              +0.005056342463750,
              +0.005056342463750,
              +0.005056342463750,
              +0.005056342463750,
              +0.005056342463750,
              +0.001786954692975,
              +0.001786954692975,
              +0.001786954692975,
              +0.001786954692975,
              +0.001786954692975,
              +0.001786954692975]

        r = F[+0.3333333333333,
              +0.5009504643520,
              -0.0019009287044,
              +0.5009504643520,
              +0.4882129579350,
              +0.0235740841305,
              +0.4882129579350,
              +0.4551366869500,
              +0.0897266360994,
              +0.4551366869500,
              +0.4019962593180,
              +0.1960074813630,
              +0.4019962593180,
              +0.2558929097590,
              +0.4882141804810,
              +0.2558929097590,
              +0.1764882559950,
              +0.6470234880100,
              +0.1764882559950,
              +0.1041708553370,
              +0.7916582893260,
              +0.1041708553370,
              +0.0530689638409,
              +0.8938620723180,
              +0.0530689638409,
              +0.0416187151960,
              +0.9167625696080,
              +0.0416187151960,
              +0.0115819214068,
              +0.9768361571860,
              +0.0115819214068,
              +0.6064026461060,
              +0.0487415836648,
              +0.3448557702290,
              +0.0487415836648,
              +0.3448557702290,
              +0.6064026461060,
              +0.6158426144570,
              +0.0063141159486,
              +0.3778432695950,
              +0.0063141159486,
              +0.3778432695950,
              +0.6158426144570,
              +0.5590480003900,
              +0.1343165205470,
              +0.3066354790620,
              +0.1343165205470,
              +0.3066354790620,
              +0.5590480003900,
              +0.7366067432630,
              +0.0139738939624,
              +0.2494193627750,
              +0.0139738939624,
              +0.2494193627750,
              +0.7366067432630,
              +0.7116751422870,
              +0.0755491329098,
              +0.2127757248030,
              +0.0755491329098,
              +0.2127757248030,
              +0.7116751422870,
              +0.8614027171550,
              -0.0083681532082,
              +0.1469654360530,
              -0.0083681532082,
              +0.1469654360530,
              +0.8614027171550,
              +0.8355869579120,
              +0.0266860632587,
              +0.1377269788290,
              +0.0266860632587,
              +0.1377269788290,
              +0.8355869579120,
              +0.9297561715570,
              +0.0105477192941,
              +0.0596961091490,
              +0.0105477192941,
              +0.0596961091490,
              +0.9297561715570]

        s = F[ +0.3333333333333,
               +0.5009504643520,
               +0.5009504643520,
               -0.0019009287044,
               +0.4882129579350,
               +0.4882129579350,
               +0.0235740841305,
               +0.4551366869500,
               +0.4551366869500,
               +0.0897266360994,
               +0.4019962593180,
               +0.4019962593180,
               +0.1960074813630,
               +0.2558929097590,
               +0.2558929097590,
               +0.4882141804810,
               +0.1764882559950,
               +0.1764882559950,
               +0.6470234880100,
               +0.1041708553370,
               +0.1041708553370,
               +0.7916582893260,
               +0.0530689638409,
               +0.0530689638409,
               +0.8938620723180,
               +0.0416187151960,
               +0.0416187151960,
               +0.9167625696080,
               +0.0115819214068,
               +0.0115819214068,
               +0.9768361571860,
               +0.3448557702290,
               +0.6064026461060,
               +0.0487415836648,
               +0.3448557702290,
               +0.6064026461060,
               +0.0487415836648,
               +0.3778432695950,
               +0.6158426144570,
               +0.0063141159486,
               +0.3778432695950,
               +0.6158426144570,
               +0.0063141159486,
               +0.3066354790620,
               +0.5590480003900,
               +0.1343165205470,
               +0.3066354790620,
               +0.5590480003900,
               +0.1343165205470,
               +0.2494193627750,
               +0.7366067432630,
               +0.0139738939624,
               +0.2494193627750,
               +0.7366067432630,
               +0.0139738939624,
               +0.2127757248030,
               +0.7116751422870,
               +0.0755491329098,
               +0.2127757248030,
               +0.7116751422870,
               +0.0755491329098,
               +0.1469654360530,
               +0.8614027171550,
               -0.0083681532082,
               +0.1469654360530,
               +0.8614027171550,
               -0.0083681532082,
               +0.1377269788290,
               +0.8355869579120,
               +0.0266860632587,
               +0.1377269788290,
               +0.8355869579120,
               +0.0266860632587,
               +0.0596961091490,
               +0.9297561715570,
               +0.0105477192941,
               +0.0596961091490,
               +0.9297561715570,
               +0.0105477192941]
    else
        w = F[]
        r = F[]
        s = F[]
    end
    return w, r, s
end
