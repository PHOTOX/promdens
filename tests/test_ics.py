import numpy as np
import pytest
from inline_snapshot import snapshot

from promdens import ENVELOPE_TYPES, InitialConditions, LaserPulse

@pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
def test_calc_field(make_pulse, envelope_type):
    pulse = make_pulse(envelope_type=envelope_type)
    ics = InitialConditions()
    ics.calc_field(pulse)

    s_time = snapshot(
        {
            "gauss": {
                0: -2.4,
                1: -1.1433629385640827,
                2: 0.11327412287183458,
                3: 1.3699111843077518,
            },
            "lorentz": {
                0: -8.0,
                1: -6.743362938564083,
                2: -5.4867258771281655,
                3: -4.230088815692248,
                4: -2.973451754256331,
                5: -1.7168146928204138,
                6: -0.4601776313844965,
                7: 0.7964594300514207,
                8: 2.053096491487338,
                9: 3.309733552923255,
                10: 4.5663706143591725,
                11: 5.82300767579509,
                12: 7.079644737231007,
            },
            "sech": {
                0: -4.4,
                1: -3.143362938564083,
                2: -1.8867258771281659,
                3: -0.6300888156922486,
                4: 0.6265482457436686,
                5: 1.8831853071795859,
                6: 3.139822368615503,
                7: 4.39645943005142,
            },
            "sin": {0: -1.0, 1: 0.25663706143591725},
            "sin2": {
                0: -1.3734125748912553,
                1: -0.1167755134553381,
                2: 1.1398615479805791,
            },
        }
    )
    for i, value in enumerate(ics.field_t):
        # NOTE: If you need to regenerate the snapshot with --inline-snapshot=fix,
        # use the strict comparison.
        assert value == s_time[envelope_type][i]
        #assert s[envelope_type][i] == pytest.approx(value, abs=1e-15)

    s_field = snapshot(
        {
            "gauss": {
                0: 0.0003307537813258052,
                1: 0.16221641712655582,
                2: 0.9823066615280812,
                3: 0.0734602147445701,
            },
            "lorentz": {
                0: 0.0065089252000307424,
                1: 0.010231866045414411,
                2: 0.016769804702885725,
                3: 0.029753410716252798,
                4: 0.06109793286196139,
                5: 0.1674681703512061,
                6: 0.7394848466234266,
                7: 0.4860163147767203,
                8: 0.12262027497972756,
                9: 0.04938586957532619,
                10: 0.025248479511786666,
                11: 0.014606587443436245,
                12: 0.009039180648695314,
            },
            "sech": {
                0: 0.000774699499234465,
                1: 0.007461030991016409,
                2: 0.07051922740592287,
                3: 0.5930372491304918,
                4: 0.5960308099790068,
                5: 0.0709643661347192,
                6: 0.007508604155336893,
                7: 0.0007796795057529445,
            },
            "sin": {0: 0.0, 1: 0.9195368430155919},
            "sin2": {0: 0.0, 1: 0.9822010377697398, 2: 0.06921814624054538},
        }
    )
    for i, value in enumerate(ics.field):
        #assert value == s_field[envelope_type][i]
        assert s_field[envelope_type][i] == pytest.approx(value, abs=1e-15)

    s_ft = snapshot(
        {
            "gauss": {
                0: 1.0,
                1: 0.9968674372922636,
                2: 0.9875504499212827,
                3: 0.9722889795455406,
                4: 0.9514758929650852,
                5: 0.9256470024888881,
                6: 0.8954682163245389,
                7: 0.8617208689432883,
                8: 0.8252864502456931,
                9: 0.7871319627370328,
                10: 0.7482968673285757,
                11: 0.7098818454651312,
                12: 0.6730381722358451,
                13: 0.6389541865319085,
                14: 0.6088323649068643,
                15: 0.5838480398374633,
                16: 0.5650815861940688,
                17: 0.5534235597891572,
                18: 0.5494677650162769,
            },
            "lorentz": {
                0: 1.0,
                1: 0.9843544172787753,
                2: 0.908451383178596,
                3: 0.8313747255891621,
                4: 0.762020790290382,
                5: 0.697723337461829,
                6: 0.6390581281941244,
                7: 0.5849716200936295,
                8: 0.5355068295493626,
                9: 0.4900106133602108,
                10: 0.4484120473358932,
                11: 0.41025919196467114,
                12: 0.37547298869123025,
                13: 0.34374498405981696,
                14: 0.3150357300503557,
                15: 0.2891677324419311,
                16: 0.2661759085293551,
                17: 0.24602093143114337,
                18: 0.22883693081842252,
                19: 0.214721865942089,
                20: 0.20389184887139586,
                21: 0.19651788879170282,
                22: 0.19278583249056677,
            },
            "sech": {
                0: 1.0,
                1: 0.9930722665761872,
                2: 0.972699739850276,
                3: 0.940054069093307,
                4: 0.8969024457185772,
                5: 0.8453779637885669,
                6: 0.7877366260144673,
                7: 0.72614541641273,
                8: 0.6625288185982675,
                9: 0.5984819799608184,
                10: 0.5352440439710395,
                11: 0.4737177881511395,
                12: 0.4145212134681439,
                13: 0.3580615219309469,
                14: 0.3046313381868895,
                15: 0.25454377528019073,
                16: 0.20835709010754647,
                17: 0.167313052803687,
                18: 0.1342111929600334,
                19: 0.11450981764994415,
            },
            "sin": {
                0: 1.0,
                1: 0.9976598025191638,
                2: 0.9906863908375584,
                3: 0.9792218876146825,
                4: 0.9635051083670007,
                5: 0.9438744989085571,
                6: 0.9207722356724137,
                7: 0.8947493332576992,
                8: 0.8664713061776268,
                9: 0.8367233625180649,
                10: 0.8064131326110722,
                11: 0.776567447056291,
                12: 0.7483177400829282,
                13: 0.7228668037497601,
                14: 0.7014292584637727,
                15: 0.6851415801292119,
                16: 0.6749469818068108,
                17: 0.6714748765312816,
            },
            "sin2": {
                0: 1.0,
                1: 0.9994500962027021,
                2: 0.9978180058537418,
                3: 0.9951560316937953,
                4: 0.9915494944600358,
                5: 0.987114017720024,
                6: 0.981991845875067,
                7: 0.9763473101672151,
                8: 0.9703615842594278,
                9: 0.9642268934855083,
                10: 0.9581403595842529,
                11: 0.9522976751730111,
                12: 0.9468868090199026,
                13: 0.9420819440592948,
                14: 0.9380378448368119,
                15: 0.9348848395069715,
                16: 0.9327245835892716,
                17: 0.9316267485688859,
            },
        }
    )

    for i, value in enumerate(ics.field_ft):
        # NOTE: If you need to regenerate the snapshot with --inline-snapshot=fix,
        # use the strict comparison.
        # assert value == s_ft[envelope_type][i]
        assert s_ft[envelope_type][i] == pytest.approx(value, abs=1e-15)

    s_ft_omega = snapshot(
        {
            "gauss": {
                0: 0.0,
                1: 0.1388888888888889,
                2: 0.2777777777777778,
                3: 0.41666666666666663,
                4: 0.5555555555555556,
                5: 0.6944444444444444,
                6: 0.8333333333333333,
                7: 0.9722222222222222,
                8: 1.1111111111111112,
                9: 1.25,
                10: 1.3888888888888888,
                11: 1.5277777777777777,
                12: 1.6666666666666665,
                13: 1.8055555555555554,
                14: 1.9444444444444444,
                15: 2.0833333333333335,
                16: 2.2222222222222223,
                17: 2.361111111111111,
                18: 2.5,
            },
            "lorentz": {
                0: 0.0,
                1: 0.11111111111111112,
                2: 0.22222222222222224,
                3: 0.3333333333333333,
                4: 0.4444444444444445,
                5: 0.5555555555555556,
                6: 0.6666666666666666,
                7: 0.7777777777777778,
                8: 0.888888888888889,
                9: 1.0,
                10: 1.1111111111111112,
                11: 1.2222222222222223,
                12: 1.3333333333333333,
                13: 1.4444444444444446,
                14: 1.5555555555555556,
                15: 1.666666666666667,
                16: 1.777777777777778,
                17: 1.888888888888889,
                18: 2.0,
                19: 2.1111111111111116,
                20: 2.2222222222222223,
                21: 2.3333333333333335,
                22: 2.4444444444444446,
            },
            "sech": {
                0: 0.0,
                1: 0.12820512820512822,
                2: 0.25641025641025644,
                3: 0.38461538461538464,
                4: 0.5128205128205129,
                5: 0.6410256410256411,
                6: 0.7692307692307693,
                7: 0.8974358974358975,
                8: 1.0256410256410258,
                9: 1.153846153846154,
                10: 1.2820512820512822,
                11: 1.4102564102564104,
                12: 1.5384615384615385,
                13: 1.6666666666666665,
                14: 1.794871794871795,
                15: 1.9230769230769234,
                16: 2.0512820512820515,
                17: 2.1794871794871793,
                18: 2.307692307692308,
                19: 2.435897435897436,
            },
            "sin": {
                0: 0.0,
                1: 0.14705882352941177,
                2: 0.29411764705882354,
                3: 0.4411764705882353,
                4: 0.5882352941176471,
                5: 0.7352941176470589,
                6: 0.8823529411764706,
                7: 1.0294117647058822,
                8: 1.1764705882352942,
                9: 1.323529411764706,
                10: 1.4705882352941178,
                11: 1.6176470588235297,
                12: 1.7647058823529411,
                13: 1.9117647058823528,
                14: 2.0588235294117645,
                15: 2.2058823529411766,
                16: 2.3529411764705883,
                17: 2.5,
            },
            "sin2": {
                0: 0.0,
                1: 0.14285714285714285,
                2: 0.2857142857142857,
                3: 0.42857142857142855,
                4: 0.5714285714285714,
                5: 0.7142857142857143,
                6: 0.8571428571428571,
                7: 1.0,
                8: 1.1428571428571428,
                9: 1.2857142857142856,
                10: 1.4285714285714286,
                11: 1.5714285714285714,
                12: 1.7142857142857142,
                13: 1.857142857142857,
                14: 2.0,
                15: 2.142857142857143,
                16: 2.2857142857142856,
                17: 2.4285714285714284,
            },
        }
    )

    for i, value in enumerate(ics.field_ft_omega):
        # NOTE: If you need to regenerate the snapshot with --inline-snapshot=fix,
        # use the strict comparison.
        # assert value == s_ft_omega[envelope_type][i]
        assert s_ft_omega[envelope_type][i] == pytest.approx(value, abs=1e-15)
