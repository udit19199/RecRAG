Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=759f88a1b31e099b ===
Gold sentences and facts:
Sentence 0: Dual - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) Age of Child Housing Food Transport . Clothing Health Child Care / Education Misc . Total Before - tax income : Less than $59,410 ( Average = $38,000 ) 0 to 2 2,990 1,160 1,170 640 630 2,040 420 9,050 3 to 5 2,990 1,260 1,230 500 590 1,910 620 9,100 6 to 8 2,990 1,710 1,350 570 660 1,290 630 8,760 9 to 11 2,990 1,970 1,350 580 710 1,910 630 9,520 12 to 14 2,990 2,130 1,480 690 1,090 1,110 700 9,960 15 to 17 2,990 2,120 1,630 730 1,010 1,290 580 9,970 Total 53,820 31,050 24,630 11,130 14,070 23,640 10,740 169,080 Before - tax income : $59,410 to $102,870 ( Average = $79,940 ) 0 to 2 3,920 1,405 1,690 760 850 2,860 890 12,370 3 to 5 3,920 1,490 1,740 610 800 2,740 1,090 12,390 6 to 8 3,920 2,100 1,860 680 940 1,680 1,110 12,290 9 to 11 3,920 2,400 1,870 710 1,000 2,110 1,100 13,110 12 to 14 3,920 2,580 1,990 840 1,410 1,910 1,170 13,820 15 to 17 3,920 2,570 2,150 900 1,330 2,400 1,050 14,320 Total 70,560 37,620 33,900 13,500 18,990 41,100 19,230 234,900 Before - tax income : More than $102,870 ( Average = $180,040 ) 0 to 2 7,100 1,900 2,550 1,050 980 5,090 1,790 20,460 3 to 5 7,100 2,000 2,610 880 930 4,970 1,990 20,480 6 to 8 7,100 2,630 2,730 970 1,080 3,910 2,000 20,420 9 to 11 7,100 2,980 2,730 1,010 1,150 4,350 2,000 21,320 12 to 14 7,100 3,190 2,860 1,170 1,610 4,700 2,070 22,700 15 to 17 7,100 3,180 3,020 1,280 1,520 6,460 1,950 24,510 Total 127,800 47,640 49,500 19,080 21,810 88,440 35,400 389,670
  Fact 0.0: The table gives USDA average spending per child for dual-parent families, not the basic cost of raising a child
  Fact 0.1: Spending categories are housing, food, transport, clothing, health, child care and education, miscellaneous, and total
  Fact 0.2: Dual-parent families with before-tax income under $59,410 have an average before-tax income of $38,000
  Fact 0.3: Dual-parent families with before-tax income under $59,410 and a child aged 0 to 2 spend $2,990 on housing, $1,160 on food, $1,170 on transport, $640 on clothing, $630 on health, $2,040 on child care and education, $420 on miscellaneous, totaling $9,050
  Fact 0.4: Dual-parent families with before-tax income under $59,410 and a child aged 3 to 5 spend $2,990 on housing, $1,260 on food, $1,230 on transport, $500 on clothing, $590 on health, $1,910 on child care and education, $620 on miscellaneous, totaling $9,100
  Fact 0.5: Dual-parent families with before-tax income under $59,410 and a child aged 6 to 8 spend $2,990 on housing, $1,710 on food, $1,350 on transport, $570 on clothing, $660 on health, $1,290 on child care and education, $630 on miscellaneous, totaling $8,760
  Fact 0.6: Dual-parent families with before-tax income under $59,410 and a child aged 9 to 11 spend $2,990 on housing, $1,970 on food, $1,350 on transport, $580 on clothing, $710 on health, $1,910 on child care and education, $630 on miscellaneous, totaling $9,520
  Fact 0.7: Dual-parent families with before-tax income under $59,410 and a child aged 12 to 14 spend $2,990 on housing, $2,130 on food, $1,480 on transport, $690 on clothing, $1,090 on health, $1,110 on child care and education, $700 on miscellaneous, totaling $9,960
  Fact 0.8: Dual-parent families with before-tax income under $59,410 and a child aged 15 to 17 spend $2,990 on housing, $2,120 on food, $1,630 on transport, $730 on clothing, $1,010 on health, $1,290 on child care and education, $580 on miscellaneous, totaling $9,970
  Fact 0.9: Dual-parent families with before-tax income under $59,410 have totals across all age groups of $53,820 housing, $31,050 food, $24,630 transport, $11,130 clothing, $14,070 health, $23,640 child care and education, $10,740 miscellaneous, and $169,080 overall
  Fact 0.10: Dual-parent families with before-tax income of $59,410 to $102,870 have an average before-tax income of $79,940
  Fact 0.11: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 0 to 2 spend $3,920 on housing, $1,405 on food, $1,690 on transport, $760 on clothing, $850 on health, $2,860 on child care and education, $890 on miscellaneous, totaling $12,370
  Fact 0.12: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 3 to 5 spend $3,920 on housing, $1,490 on food, $1,740 on transport, $610 on clothing, $800 on health, $2,740 on child care and education, $1,090 on miscellaneous, totaling $12,390
  Fact 0.13: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 6 to 8 spend $3,920 on housing, $2,100 on food, $1,860 on transport, $680 on clothing, $940 on health, $1,680 on child care and education, $1,110 on miscellaneous, totaling $12,290
  Fact 0.14: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 9 to 11 spend $3,920 on housing, $2,400 on food, $1,870 on transport, $710 on clothing, $1,000 on health, $2,110 on child care and education, $1,100 on miscellaneous, totaling $13,110
  Fact 0.15: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 12 to 14 spend $3,920 on housing, $2,580 on food, $1,990 on transport, $840 on clothing, $1,410 on health, $1,910 on child care and education, $1,170 on miscellaneous, totaling $13,820
  Fact 0.16: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 15 to 17 spend $3,920 on housing, $2,570 on food, $2,150 on transport, $900 on clothing, $1,330 on health, $2,400 on child care and education, $1,050 on miscellaneous, totaling $14,320
  Fact 0.17: Dual-parent families with before-tax income of $59,410 to $102,870 have totals across all age groups of $70,560 housing, $37,620 food, $33,900 transport, $13,500 clothing, $18,990 health, $41,100 child care and education, $19,230 miscellaneous, and $234,900 overall
  Fact 0.18: Dual-parent families with before-tax income over $102,870 have an average before-tax income of $180,040
  Fact 0.19: Dual-parent families with before-tax income over $102,870 and a child aged 0 to 2 spend $7,100 on housing, $1,900 on food, $2,550 on transport, $1,050 on clothing, $980 on health, $5,090 on child care and education, $1,790 on miscellaneous, totaling $20,460
  Fact 0.20: Dual-parent families with before-tax income over $102,870 and a child aged 3 to 5 spend $7,100 on housing, $2,000 on food, $2,610 on transport, $880 on clothing, $930 on health, $4,970 on child care and education, $1,990 on miscellaneous, totaling $20,480
  Fact 0.21: Dual-parent families with before-tax income over $102,870 and a child aged 6 to 8 spend $7,100 on housing, $2,630 on food, $2,730 on transport, $970 on clothing, $1,080 on health, $3,910 on child care and education, $2,000 on miscellaneous, totaling $20,420
  Fact 0.22: Dual-parent families with before-tax income over $102,870 and a child aged 9 to 11 spend $7,100 on housing, $2,980 on food, $2,730 on transport, $1,010 on clothing, $1,150 on health, $4,350 on child care and education, $2,000 on miscellaneous, totaling $21,320
  Fact 0.23: Dual-parent families with before-tax income over $102,870 and a child aged 12 to 14 spend $7,100 on housing, $3,190 on food, $2,860 on transport, $1,170 on clothing, $1,610 on health, $4,700 on child care and education, $2,070 on miscellaneous, totaling $22,700
  Fact 0.24: Dual-parent families with before-tax income over $102,870 and a child aged 15 to 17 spend $7,100 on housing, $3,180 on food, $3,020 on transport, $1,280 on clothing, $1,520 on health, $6,460 on child care and education, $1,950 on miscellaneous, totaling $24,510
  Fact 0.25: Dual-parent families with before-tax income over $102,870 have totals across all age groups of $127,800 housing, $47,640 food, $49,500 transport, $19,080 clothing, $21,810 health, $88,440 child care and education, $35,400 miscellaneous, and $389,670 overall

Retrieved items:
<item 1>
Cost of raising a child
880 3,080 2,370 650 1,560 4,150 1,980 19,670 15 to 17 5,880 3,220 2,370 730 1,550 5,010 1,810 20,570 Total 105,840 48,390 40,140 10,830 22,410 72,810 33,000
333,420 All numbers are in US dollars . India ( edit ) Based on an estimate by Economic Times in April 2011 , the cost of raising a child from birth to age of
majority ( 21 Years ) for a middle to upper - middle income family comes to about ₹ 55 lakh ( US $87,000 ) in total . Cost break up is as follows : Cost of
raising a child in India ( as of April , 2011 ) Expenditure Class Estimated Cost ( Rs .
</item 1>

<item 2>
Cost of raising a child
www.lv.com . Retrieved 2016 - 06 - 27 . Jump up ^ Friday , H. Swint ( July 20 , 2007 ) . `` Cost of raising children not as high as government would have you
believe '' . caller.com . Retrieved March 31 , 2013 . Jump up ^ http://abcnews.go.com/Business/page?id=4019746 ^ Jump up to : Lino , Mark . `` Expenditures on
Children by Families , 2011 ( Miscellaneous Publication Number 1528 - 2011 ) '' ( PDF ) . United States Department of Agriculture , Center for Nutrition Policy
and Promotion . Retrieved August 5 , 2012 . Jump up ^ `` How much it cost to raise a child '' ( pdf ) . report . Economic Times . 18 April 2011 . Retrieved 13
July 2016 . External links ( edit ) `` Raising that ' 07 baby will cost $204,060 in U.S. '' - from Reuters , retrieved August 5 , 2012 .
</item 2>

<item 3>
Cost of raising a child
970 1,350 580 710 1,910 630 9,520 12 to 14 2,990 2,130 1,480 690 1,090 1,110 700 9,960 15 to 17 2,990 2,120 1,630 730 1,010 1,290 580 9,970 Total 53,820 31,050
24,630 11,130 14,070 23,640 10,740 169,080 Before - tax income : $59,410 to $102,870 ( Average = $79,940 ) 0 to 2 3,920 1,405 1,690 760 850 2,860 890 12,370 3
to 5 3,920 1,490 1,740 610 800 2,740 1,090 12,390 6 to 8 3,920 2,100 1,860 680 940 1,680 1,110 12,290 9 to 11 3,920 2,
</item 3>

<item 4>
The cost of raising a child varies by country, income, family structure, and household needs; estimates cover essentials such as housing, food, clothing,
transport, health, and childcare. UNICEF-based estimates put developing-country costs at about US$900 annually, or $16,200 through age 17. UK estimates put
birth-to-21 costs at £231,843. US USDA estimates through age 18 range from $157,410–$333,420 for single parents and $
</item 4>

<item 5>
Cost of raising a child
480 6 to 8 7,100 2,630 2,730 970 1,080 3,910 2,000 20,420 9 to 11 7,100 2,980 2,730 1,010 1,150 4,350 2,000 21,320 12 to 14 7,100 3,190 2,860 1,170 1,610 4,700
2,070 22,700 15 to 17 7,100 3,180 3,020 1,280 1,520 6,460 1,950 24,510 Total 127,800 47,640 49,500 19,080 21,810 88,440 35,400 389,670 All numbers are in US
dollars . Also notice that the dual parent and solo parent charts use different income ranges so that direct comparison between the two should be undertaken
with care .
</item 5>

<item 6>
Cost of raising a child
This statistic includes children . Based on UNICEF statistics , in developing countries the cost is roughly US $ 900 for raising a child for a year , and US
$16,200 for raising a child from birth to age 17 . ( This calculation is difficult , since families in developing countries often do not operate with currency ,
but barter or trade to provide for their children . ) Half of all children in the world live in poverty . United Kingdom ( edit ) The annual LV = ( Liverpool
Victoria ) Cost of a Child report calculates the cost of raising a child from birth to 21 years old . The latest report puts the cost as £ 231,843 . The Cost of
a child calculations , from birth to 21 years , were compiled by the Centre of Economic and Business Research ( CEBR ) for LV = in December 2015 and are based
on the cost for the 21 - year period to December 2015 . Additional research was conducted by Opinium Research LLP from 22 to 27 January 2016 . The total sample
size was 1,000 UK adults with children under the age of 18 and was conducted online .
</item 6>

<item 7>
Cost of raising a child
Cost of raising a child - wikipedia Cost of raising a child Jump to : navigation , search The cost of raising a child varies from country to country . The cost
of raising a child is usually determined according to a formula that accounts for major areas of expenditure , such as food , housing , and clothing . However ,
any given family 's actual expenses may differ from the estimates . For example , the rent on a home does not usually change when the tenants have another child
, so the family 's housing costs may remain the same . In other cases , the home may be too small , in which case the family might move to a larger home at a
higher cost . Contents ( hide ) 1 Developing countries 2 United Kingdom 3 United States 3.1 Dual - Parent Family - USDA Average Spending per Child ( not the
basic cost of raising ) 3.2 Single - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) 4 India 5 See also 6 References 7
External links Developing countries ( edit ) According to Globalissues.org , `` Almost half the world -- over three billion people -- live on less than $2.50 a
day . ''
</item 7>

<item 8>
Cost of raising a child
110 12,290 9 to 11 3,920 2,400 1,870 710 1,000 2,110 1,100 13,110 12 to 14 3,920 2,580 1,990 840 1,410 1,910 1,170 13,820 15 to 17 3,920 2,570 2,150 900 1,330
2,400 1,050 14,320 Total 70,560 37,620 33,900 13,500 18,990 41,100 19,230 234,900 Before - tax income : More than $102,870 ( Average = $180,040 ) 0 to 2 7,100
1,900 2,550 1,050 980 5,090 1,790 20,460 3 to 5 7,100 2,000 2,610 880 930 4,970 1,990 20,480 6 to 8 7,100 2,630 2,
</item 8>

<item 9>
Cost of raising a child
Results have been weighted to nationally representative criteria United States ( edit ) The tax exemption per child is $3,900 in 2013 which is based on the
basic cost of raising a child . Plus there is a tax credit up to $1000 for each child. citation needed Based on a survey by the U.S. Department of Agriculture ,
the table below shows the estimated Average Spending on Children by Families . The data comes from the Consumer Expenditure Survey by the U.S. Department of
Labor , conducted from 2005 - 06 . The figures have been updated to 2011 dollars using the Consumer Price Index . However , some dispute the numbers as being
biased high for political reasons ( e.g. , Texas A&M University Finance Professor H. Swint Friday : `` The numbers , reported by the U.S. Department of
Agriculture , are outrageously misleading . Often government statistics are produced for political objectives that cause the research methodology to be biased
toward finding the highest dollar amount to support the objective . '' ) . These figures from the USDA go up to age 18 , and do not include any college or
university education .
</item 9>

<item 10>
Cost of raising a child
440 15 to 17 2,840 2,270 1,130 460 930 880 670 9,180 Total 51,120 33,090 17,850 7,080 12,840 22,980 12,450 157,410 Before - tax income : $59,410 or more (
Average = $107,820 ) 0 to 2 5,880 2,080 1,920 590 980 3,670 1,650 16,770 3 to 5 5,880 2,070 2,160 500 1,090 4,210 1,750 17,660 6 to 8 5,880 2,680 2,260 530
1,180 3,350 1,930 17,810 9 to 11 5,880 3,000 2,300 610 1,110 3,880 1,880 18,660 12 to 14 5,880 3,080 2,370 650 1,560 4,
</item 10>
=== END ITEM id=759f88a1b31e099b ===

=== ITEM id=44a66540d9690944 ===
Gold sentences and facts:
Sentence 0: The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert .
  Fact 0.0: The Second Great Awakening occurred in several episodes.
  Fact 0.1: The Second Great Awakening occurred over different denominations.
  Fact 0.2: The revivals of the Second Great Awakening were very similar to each other.
  Fact 0.3: Revival meetings were the most effective form of evangelizing during this period.
  Fact 0.4: Revival meetings cut across geographical boundaries.
  Fact 0.5: The revival movement quickly spread throughout Kentucky, Tennessee and southern Ohio.
  Fact 0.6: Each denomination had assets that allowed it to thrive on the frontier.
  Fact 0.7: The Methodists had an efficient organization.
  Fact 0.8: The Methodist organization depended on itinerant ministers known as circuit riders.
  Fact 0.9: Circuit riders sought out people in remote frontier locations.
  Fact 0.10: Circuit riders came from among the common people.
  Fact 0.11: Coming from among the common people helped circuit riders establish rapport with the frontier families they hoped to convert.
Sentence 1: The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ .
  Fact 1.0: The revivals enrolled millions of new members in existing evangelical denominations.
  Fact 1.1: The revivals led to the formation of new denominations.
  Fact 1.2: Many converts believed that the Awakening heralded a new millennial age.
  Fact 1.3: The Second Great Awakening stimulated the establishment of many reform movements.
  Fact 1.4: The reform movements were designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ.
Sentence 2: In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be saved was from grace from God .
  Fact 2.0: In the newly settled frontier regions, the revival was implemented through camp meetings.
  Fact 2.1: Camp meetings often provided the first encounter for some settlers with organized religion.
  Fact 2.2: Camp meetings were important as social venues.
  Fact 2.3: A camp meeting was a religious service of several days' length with preachers.
  Fact 2.4: Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship.
  Fact 2.5: Participating in a religious revival with crowds of hundreds and perhaps thousands of people was sheer exhilaration.
  Fact 2.6: That exhilaration inspired the dancing, shouting, and singing associated with camp meeting events.
  Fact 2.7: The revivals followed an arc of great emotional power.
  Fact 2.8: The revivals emphasized the individual's sins and need to turn to Christ.
  Fact 2.9: The revivals were restored by a sense of personal salvation.
  Fact 2.10: Upon their return home, most converts joined or created small local churches.
  Fact 2.11: The small local churches grew rapidly.
  Fact 2.12: The Second Great Awakening marked a religious transition in society in America.
  Fact 2.13: Many Americans from the Calvinist sect emphasized man's inability to save themselves.
  Fact 2.14: Many Americans from the Calvinist sect emphasized that their only way to be saved was by grace from God.

Retrieved items:
<item 1>
Second Great Awakening
a re-emphasis on Wesleyan teachings on sanctification emerged during the Second Great Awakening , leading to a distinction between Mainline Methodism and
Holiness churches . Restoration movement ( edit ) Main article : Restoration Movement See also : Restorationism The idea of restoring a `` primitive '' form of
Christianity grew in popularity in the U.S. after the American Revolution . This desire to restore a purer form of Christianity without an elaborate hierarchy
contributed to the development of many groups during the Second Great Awakening , including the Mormons , Baptists and Shakers . Several factors made the
restoration sentiment particularly appealing during this time period : To immigrants in the early 19th century , the land in the United States seemed pristine ,
edenic and undefiled -- `` the perfect place to recover pure , uncorrupted and original Christianity '' -- and the tradition - bound European churches seemed
out of place in this new setting . A primitive faith based on the Bible alone promised a way to sidestep the competing claims of the many denominations
available and for congregations to find assurance of being right without the security of an established national church . The Restoration Movement began during
, and was greatly influenced by , the Second Great Awakening .
</item 1>

<item 2>
Second Great Awakening
West ( edit ) Main article : Revival of 1800 In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided
the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several
days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of
participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these
events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of
personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a
religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be
saved was from grace from God . The Revival of 1800 in Logan County , Kentucky , began as a traditional Presbyterian sacramental occasion .
</item 2>

<item 3>
Second Great Awakening
Contents ( hide ) 1 Spread of revivals 1.1 Background 1.2 Theology 1.3 Burned - over district 1.4 West and Tidewater South 1.5 West 1.6 Church membership soars
2 Subgroups 2.1 Adventism 2.2 Holiness movement 2.3 Restoration Movement 3 Culture and society 4 Slaves and free Africans 5 Women 6 Prominent figures 7
Political implications 8 See also 9 References 10 Further reading 10.1 Historiography Spread of revivals ( edit ) Part of a series on Baptists Background ( show
) Christianity Protestantism Puritanism Anabaptism Doctrine ( show ) Priesthood of all believers Individual soul liberty Separation of church and state Sola
scriptura Congregationalism Ordinances Offices Confessions Key figures ( show ) John Smyth Thomas Helwys Roger Williams John Clarke John Bunyan Shubal Stearns
Andrew Fuller Charles Spurgeon D.N. Jackson James Robinson Graves William Bullein Johnson William Carey Luther Rice Martin Luther King Jr .
</item 3>

<item 4>
Second Great Awakening
Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements
designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ . Historians named the Second Great Awakening in the context of the
First Great Awakening of the 1730s and 1740s and of the Third Great Awakening of the late 1850s to early 1900s . These revivals were part of a much larger
Romantic religious movement that was sweeping across Europe at the time , mainly throughout England , Scotland , and Germany .
</item 4>

<item 5>
Second Great Awakening
See Uncommon Christian Ministries Long , Kimberly Bracken . `` The Communion Sermons of James Mcgready : Sacramental Theology and Scots - Irish Piety on the
Kentucky Frontier '' , Journal of Presbyterian History , 2002 80 ( 1 ) : 3 -- 16 . ISSN 0022 - 3883 . JSTOR 23336302 . Loveland Anne C. Southern Evangelicals
and the Social Order , 1800 -- 1860 , ( 1980 ) McLoughlin William G. Modern Revivalism , 1959 . McLoughlin William G. Revivals , Awakenings , and Reform : An
Essay on Religion and Social Change in America , 1607 -- 1977 , 1978 . Marsden , George M. The Evangelical Mind and the New School Presbyterian Experience : A
Case Study of Thought and Theology in Nineteenth - Century America ( 1970 ) . Meyer , Neil . `` Falling for the Lord : Shame , Revivalism , and the Origins of
the Second Great Awakening . ''
</item 5>

<item 6>
Second Great Awakening
the Second reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skepticism , deism , and
rationalism left over from the Enlightenment . At about the same time , similar movements flourished in Europe . Pietism was sweeping German countries .
Evangelicalism was waxing strong in England . The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals
were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement
quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had
an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit
riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert . Theology ( edit ) Main article
: Postmillennialism Postmillennialism theology dominated American Protestantism in the first half of the 19th century .
</item 6>

<item 7>
Second Great Awakening
The Second Great Awakening served as an `` organizing process '' that created `` a religious and educational infrastructure '' across the western frontier that
encompassed social networks , a religious journalism that provided mass communication , and church - related colleges . Publication and education societies
promoted Christian education ; most notable among them was the American Bible Society , founded in 1816 . Women made up a large part of these voluntary
societies . The Female Missionary Society and the Maternal Association , both active in Utica , NY , were highly organized and financially sophisticated women
's organizations responsible for many of the evangelical converts of the New York frontier . There were also societies that broadened their focus from
traditional religious concerns to larger societal ones . These organizations were primarily sponsored by affluent women . They did not stem entirely from the
Second Great Awakening , but the revivalist doctrine and the expectation that one 's conversion would lead to personal action accelerated the role of women 's
social benevolence work . Social activism influenced abolition groups and supporters of the Temperance movement . They began efforts to reform prisons and care
for the handicapped and mentally ill . They believed in the perfectibility of people and were highly moralistic in their endeavors .
</item 7>

<item 8>
Second Great Awakening
Second Great Awakening - wikipedia Second Great Awakening Jump to : navigation , search An engraving of a Methodist camp meeting in 1819 ( Library of Congress )
. Great Awakening First ( c. 1730 -- 1755 ) Second ( c. 1790 -- 1840 ) Third ( c. 1855 -- 1900 ) Fourth ( c. 1960 -- 1980 ) The Second Great Awakening was a
Protestant religious revival during the early 19th century in the United States . The movement began around 1790 , gained momentum by 1800 and , after 1820 ,
membership rose rapidly among Baptist and Methodist congregations whose preachers led the movement . It was past its peak by the late 1850s . The Second Great
Awakening reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skeptical rationalism and deism of
the Enlightenment . The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many
converts believed that the Awakening heralded a new millennial age .
</item 8>

<item 9>
Second Great Awakening
While Protestant religion had previously played an important role on the American political scene , the Second Great Awakening strengthened the role it would
play . See also ( edit ) Baptist portal Methodism portal Calvinism portal Seventh - day Adventist Church portal Latter Day Saints portal LDS Church portal
Advent Christian Church Christian revival Christianity in the 19th century The Church of Jesus Christ of Latter - day Saints Cumberland Presbyterian Church
Ethnocultural politics in the United States Holiness movement Restoration Movement Seventh - day Adventist Church References ( edit ) ^ Jump up to : Timothy L.
Smith , Revivalism and Social Reform : American Protestantism on the Eve of the Civil War ( 1957 ) . Jump up ^ Heyrman , Christine Leigh . `` The First Great
Awakening '' . Divining America , TeacherServe . National Humanities Center . Jump up ^ Henry B. Clark ( 1982 ) . Freedom of Religion in America : Historical
Roots , Philosophical Concepts , Contemporary Problems . Transaction Publishers . p. 16 . ^ Jump up to : Nancy Cott , `` Young Women in the Second Great
Awakening in New England , '' Feminist Studies , ( 1975 ) , 3 # 1 p.
</item 9>

<item 10>
Second Great Awakening
The Restoration Movement began during , and was greatly influenced by , the Second Great Awakening . While the leaders of one of the two primary groups making
up this movement , Thomas Campbell and Alexander Campbell , resisted what they saw as the spiritual manipulation of the camp meetings , the revivals contributed
to the development of the other major branch , led by Barton W. Stone . The Southern phase of the Awakening `` was an important matrix of Barton Stone 's reform
movement '' and shaped the evangelistic techniques used by both Stone and the Campbells . Culture and society ( edit ) Efforts to apply Christian teaching to
the resolution of social problems presaged the Social Gospel of the late 19th century . Converts were taught that to achieve salvation they needed not just to
repent personal sin but also work for the moral perfection of society , which meant eradicating sin in all its forms . Thus , evangelical converts were leading
figures in a variety of 19th century reform movements . Congregationalists set up missionary societies to evangelize the western territory of the northern tier
. Members of these groups acted as apostles for the faith , and also as educators and exponents of northeastern urban culture .
</item 10>
=== END ITEM id=44a66540d9690944 ===

=== ITEM id=43a36c1d4acfac3e ===
Gold sentences and facts:
Sentence 0: The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as well as the day - to - day application of such rules . As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside of confederation has not adopted them but instead has its own Código de Ética Profesional .
  Fact 0.0: The ABA Model Rules of Professional Conduct were created by the American Bar Association (ABA).
  Fact 0.1: The ABA Model Rules of Professional Conduct are a set of rules.
  Fact 0.2: The ABA Model Rules prescribe baseline standards of legal ethics for lawyers in the United States.
  Fact 0.3: The ABA Model Rules prescribe baseline standards of professional responsibility for lawyers in the United States.
  Fact 0.4: The ABA House of Delegates promulgated the rules upon the recommendation of the Kutak Commission.
  Fact 0.5: The Kutak Commission's recommendation came in 1983.
  Fact 0.6: The rules are merely recommendations, or models.
  Fact 0.7: The name Model Rules reflects that the rules are recommendations or models.
  Fact 0.8: The rules are not themselves binding.
  Fact 0.9: Having a common set of Model Rules facilitates a common discourse on legal ethics.
  Fact 0.10: Having a common set of Model Rules simplifies professional responsibility training.
  Fact 0.11: Having a common set of Model Rules simplifies the day-to-day application of such rules.
  Fact 0.12: As of 2015, 49 states and four territories had adopted the rules in whole or in part.
  Fact 0.13: The most recent jurisdiction to adopt the rules as of 2015 was the Commonwealth of the Northern Mariana Islands.
  Fact 0.14: The Commonwealth of the Northern Mariana Islands adopted the rules in March 2015.
  Fact 0.15: California is the only state that has not adopted the ABA Model Rules.
  Fact 0.16: Puerto Rico is the only U.S. jurisdiction outside of confederation that has not adopted the ABA Model Rules.
  Fact 0.17: Puerto Rico has its own Codigo de Etica Profesional instead.

Retrieved items:
<item 1>
American Bar Association Model Rules of Professional Conduct
However , the Model Rules have been adopted , in whole or in part , and sometimes in variation , as the rules of professional conduct for attorneys in 49 states
, the District of Columbia , and four of the five inhabited U.S. territories . The rules of professional conduct or professional responsibility adopted in a
particular state are legally enforceable against the lawyers of that state as well as any lawyer practicing there on a pro hac vice basis . On December 17 ,
2008 , New York announced that it would finally abandon the old ABA Model Code of Professional Responsibility ( the last state to do so ) and adopt a heavily
modified version of the ABA Model Rules , effective April 1 , 2009 . New York 's version of the Model Rules was created by adjusting the standard Model Rules to
reflect indigenous New York rules that had been incorporated over the years into its version of the Model Code .
</item 1>

<item 2>
American Bar Association Model Rules of Professional Conduct
Jump up ^ http://www.mebaroverseers.org/attorney_regulation/maine_conduct_rules.html Jump up ^
http://www.maine.gov/tools/whatsnew/index.php?topic=mebar_overseers_bar_rules&id=88185&v=article Jump up ^ http://www.cnmilaw.org/pdf/court_rules/R07.pdf
Northern Mariana Islands Rules of Attorney Discipline and Procedure . ^ Jump up to : Rogers , Joan C. ( 2 October 2014 ) . `` California Justices Tell State Bar
to Redo Proposals for Updating Lawyer Conduct Rules '' . ABA / BNA Lawyers ' Manual on Professional Conduct . Bloomberg BNA . Retrieved 15 October 2014 . ^ Jump
up to : Zitrin , Richard ( 10 October 2014 ) . `` Viewpoint : State Supreme Court Resets Ethics Rewrite '' . The Recorder . ALM Media Properties , LLC .
Retrieved 15 October 2014 . Jump up ^ http://www.calbar.ca.gov/AboutUs/BoardofTrustees.aspx Board of Trustees , State Bar of California .
</item 2>

<item 3>
American Bar Association Model Rules of Professional Conduct
Even though New York did not adopt the Model Rules verbatim , the advantage of adopting its overall structure is that it simplifies the professional
responsibility training of New York lawyers , and makes it easier for out - of - state lawyers to conform their conduct to New York rules by simply comparing
their home state 's version of the Model Rules to New York 's version . In June 2009 , the Supreme Court of Maine approved the adoption of the Model Rules in
that state , effective August 1 , 2009 . Among the portions of the Model Rules that the Maine Rules of Professional Conduct do not include is Model Rule 1.8 ( j
) ( 2002 ) categorically prohibiting sexual relations between lawyer and client . In March 2015 , the Supreme Court of the Commonwealth of the Northern Mariana
Islands adopted its current Rules of Attorney Discipline and Procedure , which incorporates the ABA Model Rules by reference and expands upon them by adding
another 22 local rules of professional conduct . California ( edit ) California still has its own set of unique rules governing professional attorney conduct .
</item 3>

<item 4>
American Bar Association Model Rules of Professional Conduct
As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern
Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside
of confederation has not adopted them but instead has its own Código de Ética Profesional . Contents ( hide ) 1 State , commonwealth and territorial adoptions
of the Rules 1.1 California 2 Professional regulation 3 See also 4 References 5 External links State , Commonwealth and territorial adoptions of the Rules (
edit ) The American Bar Association is a private sector voluntary professional association . As such , it has no lawmaking power or regulatory authority , but
it is just like any other nongovernmental bar association . Accordingly , the Model Rules are not legally binding on ABA members or within any particular
jurisdiction .
</item 4>

<item 5>
American Bar Association Model Rules of Professional Conduct
California ( edit ) California still has its own set of unique rules governing professional attorney conduct . Although many of the rules are similar and
sometimes identical to the ABA Model Rules , the California rules differ markedly from the ABA models in both format , scope and content . For example , under
the ABA Model Rules , an attorney may be disciplined for the appearance of impropriety , regardless whether any actual impropriety occurred ; whereas the
California rules permit discipline for only actual impropriety . This reflects the nature of California law . It is unlikely that California will ever adopt the
ABA Model Rules apart beyond harmonising the format and existing California rules that are identical or nearly identical to ABA Model Rules . Reasons for this
reluctance include the fact there are more than 280,000 attorneys licensed in California , so a drastic change from the existing rules to completely different
rules would shock the profession and , ultimately , would most likely fail . Additionally , many of California 's rules of professional conduct are established
by statute and , thus , would require action by the California Legislature to change them ; or , alternatively , action by the people of California through the
voter initiative process .
</item 5>

<item 6>
American Bar Association Model Rules of Professional Conduct
This is due to California law and public policy being so markedly different from the rest of the country in so many ways . For example , California is one of
only four states where covenants not to compete are void as against public policy ; and it is one of only three states where trusts are presumed to be revocable
unless the trust instrument explicitly states the trust is irrevocable ( whereas the opposite is true in the other 47 states ) . By way of comparative example ,
California has long held the position that clients must be encouraged to speak candidly to their legal counsel about both past and future actions knowing their
statements will be held in complete confidence . Their counsel can then explain the relevant law to them and urge them to conform their future conduct to the
law , thus promoting lawful conduct . In contrast , ABA Model Rule 1.6 contains exceptions to confidentiality for both violent and nonviolent ( e.g. financial )
future crimes , which California believes creates a strong disincentive for clients to discuss their future plans with their attorneys - but for which the
client might chose to comply with the law instead of violating it .
</item 6>

<item 7>
American Bar Association Model Rules of Professional Conduct
External links ( edit ) The ABA Model Rules of Professional Conduct Pre-2002 Changes to the Model Rules since its adoption in 1983 from Cornell Law School
Retrieved from `` https://en.wikipedia.org/w/index.php?title=American_Bar_Association_Model_Rules_of_Professional_Conduct&oldid=789593953 '' Categories : 1983
in law Legal ethics Codes of conduct American Bar Association United States law 1983 documents Hidden categories : Articles containing potentially dated
statements from 2015 All articles containing potentially dated statements Talk Contents About Wikipedia 한국어 Suomi Edit links This page was last edited on 8 July
2017 , at 08 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
American Bar Association Model Rules of Professional Conduct
While either of their secretaries could be a member of the State Bar , they most likely are not . Similarly , the Board of Trustees that oversee the business
operations of the State Bar of California under supervision of the California Supreme Court is legally required to be a mix of attorneys licensed to practice in
California and `` public members '' who may not be licensed attorneys . So , while American attorneys are regulated by public sector agencies instead of private
sector law societies , most of the people who develop the regulatory policies are practising attorneys and sitting judges , so there is a degree of self -
regulation but not nearly to the same degree that solicitors and barristers in England and Wales , and other Commonwealth jurisdictions are self - regulated by
private sector associations . See also ( edit ) American Bar Association Model Code of Professional Responsibility American Bar Association References ( edit )
Jump up ^ Press Release : New Attorney Rules of Professional Conduct Announced , Communications Office of the New York Courts , 17 December 2008 .
</item 8>

<item 9>
American Bar Association Model Rules of Professional Conduct
American Bar Association Model Rules of professional conduct - wikipedia American Bar Association Model Rules of professional conduct Jump to : navigation ,
search Professional responsibility Duties to the client Confidentiality Avoiding conflict of interest Due diligence and Competence ( law ) Avoid commingling
Avoid self - dealing Effective assistance Avoid fee splitting Withdrawal from representation Duties to the court Disclosure of perjury Disclosure of adverse
authority Duties to the profession Limitations on legal advertising Report misconduct Sources of law ABA Model Rules Penalties for misconduct Disbarment
Judicial misconduct The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline
standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the
recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves
binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as
well as the day - to - day application of such rules .
</item 9>

<item 10>
American Bar Association Model Rules of Professional Conduct
In 2001 , the Commission for the Revision of the Rules of Professional Conduct of the State Bar of California began a comprehensive revision of the California
rules that would , among other things , convert them into a localized version of the ABA Model Rules . However , the Commission 's progress had been extremely
slow , simply because there are so many substantive and structural differences between the California rules and the ABA models . In October 2010 , the State Bar
of California formally adopted the Commission 's report , recommending adoption of 67 proposed changes to harmonise the California Rules of Professional Conduct
with the ABA Model Rules of Professional Responsibility . On September 19 , 2014 , however , for reasons that were not fully explained , the Supreme Court of
California rejected all of the proposed revisions and returned them to the State Bar . The Court 's letter directed the State Bar to start the process all over
again with a new Commission , and to submit a new set of revised rules by March 31 , 2017 . Professional regulation ( edit ) Except for the Patent Bar , there
is no national legal profession in the United States but there are 56 separate legal professions .
</item 10>
=== END ITEM id=43a36c1d4acfac3e ===

=== ITEM id=5901c7805a6e0b7a ===
Gold sentences and facts:
Sentence 0: In the aftermath of the disputed 1824 presidential election , the separate factions took on many characteristics of parties in their own right . Adams ' supporters , in league with Clay , favored modernization , banks , industrial development and federal spending for roads and other internal improvements , which the Old Republicans and the Jackson men usually opposed . Writing in his personal journal on December 13 , 1826 , President Adams noted the difficulty he faced in attempting to be nonpartisan in appointing men to office :
  Fact 0.0: After the disputed 1824 presidential election, the separate factions took on many characteristics of parties in their own right.
  Fact 0.1: Adams' supporters, in league with Clay, favored modernization.
  Fact 0.2: Adams' supporters, in league with Clay, favored banks.
  Fact 0.3: Adams' supporters, in league with Clay, favored industrial development.
  Fact 0.4: Adams' supporters, in league with Clay, favored federal spending for roads and other internal improvements.
  Fact 0.5: The Old Republicans and the Jackson men usually opposed these positions.
  Fact 0.6: President Adams wrote in his personal journal on December 13, 1826.
  Fact 0.7: President Adams noted in his personal journal on December 13, 1826 the difficulty he faced in attempting to be nonpartisan in appointing men to office.

Retrieved items:
<item 1>
Democratic-Republican Party
Foreign affairs took a leading role in 1794 -- 1795 as the Republicans vigorously opposed the Jay Treaty with Britain , which was then at war with France .
Republicans saw France as more democratic after its revolution , while Britain represented the hated monarchy . The party denounced many of Hamilton 's measures
as unconstitutional , especially the national bank . The party was strongest in the South and weakest in the Northeast . It demanded states ' rights as
expressed by the `` Principles of 1798 '' articulated in the Kentucky and Virginia Resolutions that would allow states to nullify a federal law . Above all ,
the party stood for the primacy of the yeoman farmers . Republicans were deeply committed to the principles of republicanism , which they feared were threatened
by the supposed monarchical tendencies of the Hamiltonian Federalists . The party came to power in 1801 with the election of Jefferson in the 1800 presidential
election . The Federalists -- too elitist to appeal to most people -- faded away and totally collapsed after 1815 . Despite internal divisions , the Republicans
dominated the First Party System until partisanship itself withered away during the Era of Good Feelings after 1816 .
</item 1>

<item 2>
Democratic-Republican Party
It began in 1791 as one faction in Congress and included many politicians who had been opposed to the new constitution . They called themselves `` Republicans
'' after their ideology , republicanism . They distrusted the Federalist commitment to republicanism . The party splintered in 1824 into the Jacksonian movement
( which became the Democratic Party in 1828 ) and the short - lived National Republican Party ( later succeeded by the Whig Party ) . The term `` Democratic -
Republican '' is used especially by modern political scientists for the first `` Republican Party '' ( as opposed to the modern Republican Party founded in 1854
) . It is also known as the Jeffersonian Republicans . Historians typically use the title `` Republican Party '' . An `` Anti-Administration '' faction met
secretly in the national capital ( Philadelphia ) to oppose Hamilton 's financial programs . Jefferson denounced the programs as leading to monarchy and
subversive of republicanism . Jefferson needed to have a nationwide party to challenge the Federalists , which Hamilton was building up with allies in major
cities .
</item 2>

<item 3>
Democratic-Republican Party
HuffPost . Retrieved August 30 , 2017 . Jump up ^ `` Democratic - Republican Party '' . Encyclopædia Britannica . July 20 , 1998 . Retrieved August 30 , 2017 .
The Republicans contended that the Federalists harboured aristocratic attitudes and that their policies placed too much power in the central government and
tended to benefit the affluent at the expense of the common man . Jump up ^ `` Democratic - Republican Party '' . The Gilder Lehrman Institute of American
History . Retrieved December 7 , 2014 . Jump up ^ Susan Dunn ( 2004 ) . Jefferson 's Second Revolution : The Election of 1800 and the Triumph of Republicanism .
Houghton Mifflin Harcourt . p. 279 . ISBN 0618131647 . Jump up ^ Paul Kleppner , et . al .
</item 3>

<item 4>
Democratic-Republican Party
Jefferson came in second in the electoral college ( at the time , its balloting could not distinguish between President and Vice President ) and became Vice
President . He would become a consistent and strong opponent of the policies of the John Adams administration . Jefferson and Madison were deeply upset by the
unconstitutionality of the Alien and Sedition Acts of 1798 and they secretly wrote the Kentucky and Virginia Resolutions , which called on state legislatures to
nullify unconstitutional laws . However , the other states did not follow suit and several rejected the notion that states could nullify federal law . The
Republican critique of federalism became wrapped in the slogan of `` Principles of 1798 '' , which became the hallmark of the party . The most important of
these principles were states ' rights , opposition to a strong national government , distrust of the federal courts and opposition to the navy and the national
bank . The party saw itself as a champion of republicanism and denounced the Federalists as supporters of monarchy and aristocracy . The party coalesced around
Jefferson , who diligently maintained extensive correspondence with like - minded Republican leaders throughout the country .
</item 4>

<item 5>
Democratic-Republican Party
I scarcely know myself which is most to be deprecated , a consolidation , or dissolution of the states . The horrors of both are beyond the reach of human
foresight . In the aftermath of the disputed 1824 presidential election , the separate factions took on many characteristics of parties in their own right .
Adams ' supporters , in league with Clay , favored modernization , banks , industrial development and federal spending for roads and other internal improvements
, which the Old Republicans and the Jackson men usually opposed . Writing in his personal journal on December 13 , 1826 , President Adams noted the difficulty
he faced in attempting to be nonpartisan in appointing men to office : And it is upon the occasion of appointments to office that all the wormwood and the gall
of the old party hatred ooze out . Not a vacancy to any office occurs but there is a distinguished federalist started and pushed home as a candidate to fill it
-- always well qualified , sometimes in an eminent degree , and yet so obnoxious to the Republican party that he can not be appointed without exciting a
vehement clamor against him and against the Administration .
</item 5>

<item 6>
Democratic-Republican Party
Politicians ever since have used the issue of a high national debt to denounce the other party for profligacy and a threat to fiscal soundness and the nation 's
future . Monroe and Adams ( 1816 -- 1828 ) ( edit ) In rapidly expanding western states , the Federalists had few supporters . Every state had a distinct
political geography that shaped party membership . In Pennsylvania , the Republicans were weakest around Philadelphia and strongest in Scots - Irish settlements
in the west . Members came from all social classes , but came predominantly from the poor , subsistence farmers , mechanics and tradesmen . After the War of
1812 , partisanship subsided across the young republic -- people called it the Era of Good Feelings . James Monroe narrowly won the party 's nomination for
President in Congress over William Crawford in 1816 and defeated Federalist Rufus King in the general election . In the early years of the party , the key
central organization grew out of caucuses of Congressional leaders in Washington . However , the key battles to choose electors occurred in the states , not in
the caucus .
</item 6>

<item 7>
Democratic-Republican Party
On rare occasions a Senator resigned rather than follow instructions . The opposition Federalist Party quickly declined , suffering from a lack of leadership
after the death of Hamilton and the retirement of John Adams . It revived briefly in opposition to the War of 1812 , but the extremism of its Hartford
Convention of 1815 utterly destroyed it as a political force . National debt ( edit ) Jefferson and Albert Gallatin focused on the danger that the public debt ,
unless it was paid off , would be a threat to republican values . They were appalled that Hamilton was increasing the national debt and using it to solidify his
Federalist base . Gallatin was the Republican Party 's chief expert on fiscal issues and as Treasury Secretary under Jefferson and Madison worked hard to lower
taxes and lower the debt , while at the same time paying cash for the Louisiana Purchase and funding the War of 1812 .
</item 7>

<item 8>
Democratic-Republican Party
The new party set up newspapers that made withering critiques of Hamiltonianism , extolled the yeoman farmer , argued for strict construction of the
Constitution , favored the French Revolution , strongly opposed Great Britain and called for stronger state governments than the Federalist Party was proposing
. Presidential elections of 1792 and 1796 ( edit ) The elections of 1792 were the first ones to be contested on anything resembling a partisan basis . In most
states the congressional elections were recognized , as Jefferson strategist John Beckley put it , as a `` struggle between the Treasury department and the
republican interest '' . In New York , the candidates for governor were John Jay , a Federalist ; and incumbent George Clinton , who was allied with Jefferson
and the Republicans . Four states ' electors voted for Clinton and one ( Kentucky ) for Jefferson for Vice President in opposition to incumbent John Adams as
well as casting their votes for President Washington . Before 1804 , electors cast two votes together without differentiation as to which office was to be
filled by which candidate . In the 1796 election , the party made its first bid for the presidency with Jefferson as its presidential candidate and Aaron Burr
as its vice presidential candidate .
</item 8>

<item 9>
Democratic-Republican Party
Thomas Jefferson wrote on the state of party politics in the early 1820s : An opinion prevails that there is no longer any distinction , that the republicans &
Federalists are completely amalgamated but it is not so . The amalgamation is of name only , not of principle . All indeed call themselves by the name of
Republicans , because that of Federalists was extinguished in the battle of New Orleans . But the truth is that finding that monarchy is a desperate wish in
this country , they rally to the point which they think next best , a consolidated government . Their aim is now therefore to break down the rights reserved by
the constitution to the states as a bulwark against that consolidation , the fear of which produced the whole of the opposition to the constitution at its birth
. Hence new Republicans in Congress , preaching the doctrines of the old Federalists , and the new nick - names of Ultras and Radicals . But I trust they will
fail under the new , as the old name , and that the friends of the real constitution and union will prevail against consolidation , as they have done against
monarchism .
</item 9>

<item 10>
Democratic-Republican Party
Washington frequently decried the growing sense of `` party '' emerging from the internal battles among Jefferson , Madison , Hamilton , Adams and others in his
administration . As warfare in Europe increased , the two factions increasingly made foreign policy the central political issue of the day . The Republicans
wanted to maintain the 1778 alliance with France , which had overthrown the monarchy and aristocracy and become a republic . Even though Britain was by far
United States ' leading trading partner , Republicans feared that increased trade would undermine republicanism . The Republicans distrusted Hamilton 's
national bank and rejected his premise that a national debt was good for the country as Republicans said they were both forms of corruption . They strongly
distrusted the elitism of Hamilton 's circle , denouncing it as `` aristocratic '' ; and they called for states ' rights lest the Federalists centralize ever
more power in the national governments . The intense debate over the Jay Treaty in 1794 -- 1795 transformed those opposed to Hamilton 's policies from a loose
movement into a true political party .
</item 10>
=== END ITEM id=5901c7805a6e0b7a ===

=== ITEM id=3040413436f05365 ===
Gold sentences and facts:
Sentence 0: Although Surface Pro 4 is backward compatible with Type Cover accessory of Surface Pro 3 , an attachable keyboard which doubles as a screen protector , the Surface Pro 4 Type Cover has been announced with and without a fingerprint ID sensor . The new Type Cover is thinner and lighter than its predecessor , has improved magnetic stability for lap use , a mechanical keyboard with improved key spacing , as well as a larger touchpad .
  Fact 0.0: Surface Pro 4 is backward compatible with the Type Cover accessory of Surface Pro 3.
  Fact 0.1: The Type Cover is an attachable keyboard that doubles as a screen protector.
  Fact 0.2: The Surface Pro 4 Type Cover has been announced with a fingerprint ID sensor.
  Fact 0.3: The Surface Pro 4 Type Cover has been announced without a fingerprint ID sensor.
  Fact 0.4: The new Type Cover is thinner than its predecessor.
  Fact 0.5: The new Type Cover is lighter than its predecessor.
  Fact 0.6: The new Type Cover has improved magnetic stability for lap use.
  Fact 0.7: The new Type Cover has a mechanical keyboard with improved key spacing.
  Fact 0.8: The new Type Cover has a larger touchpad.

Retrieved items:
<item 1>
Surface Pro 4
designed , developed , marketed , and produced by Microsoft . The Surface Pro 4 was announced on October 6 , 2015 alongside the Surface Book . In the U.S. and
Canada , the Surface Pro 4 was released on October 26 , 2015 . The Surface Pro 4 is an update over its predecessor , featuring Skylake CPUs , more RAM and SSD
options and a 12.3 - inch display with a greater resolution . At the same time , the device is compatible with most of its predecessor 's accessories . Contents
( hide ) 1 History 2 Features 2.1 Hardware 2.2 Software 2.3 Accessories 3 Issues 4 Reception 5 Timeline 6 References History ( edit ) The Surface Pro 4 was
announced on October 6 , 2015 alongside the Surface Book . Both devices were available for pre-order the following day and available for customers beginning on
October 26 , 2015 . The device initially exhibited failure to sleep properly , which drained the battery very quickly . Microsoft subsequently developed a fix
that was made available on February 17 , 2016 .
</item 1>

<item 2>
Surface Pro 4
Surface Pro 4 - wikipedia Surface Pro 4 Jump to : navigation , search Surface Pro 4 Developer Microsoft Product family Surface Type 2 - in - 1 detachable
Generation 4th Release date October 26 , 2015 Introductory price $899 ( £ 749 ) Operating system Windows 10 Pro CPU Intel Skylake low - voltage dual - core : m3
- 6Y30 900 MHz up to 2.2 GHz , 4 MB cache , 4.5 W i5 - 6300U 2.4 up to 3.0 GHz , 3 MB cache , 15 W i7 - 6650U 2.2 up to 3.4 GHz , 4 MB cache , 15 W Memory 4 GB
, 8 GB , or 16 GB Storage PCI - Express SSD : 128 GB , 256 GB , 512 GB , or 1 TB Display 12.
</item 2>

<item 3>
Surface Pro 4
Jump up ^ `` ARK Intel Core i5 Specifications '' . Jump up ^ `` ARK Intel Core i7 Specifications '' . ^ Jump up to : `` Surface Pro 4 -- Power Through Your Day
'' . October 6 , 2015 . Jump up ^ Surface Pro 5 release date , news and rumors Jump up ^ Microsoft Unveils the Surface Pro 4 With a 12.3 - inch Display Jump up
^ Surface Pro 4 , Surface Book ' sleep ' fix coming soon ( early 2016 ) Jump up ^ Microsoft Patches Surface Book And Surface Pro 4 Sleep Issue Jump up ^ ``
Surface Pro 4 tablet models '' . November 5 , 2015 . Jump up ^ Heslip , Ken ( 6 October 2015 ) . `` Microsoft 's new Surface Pen now available for pre-order ''
. WinBeta . Jump up ^ Guim , Mark ( 22 October 2015 ) . `` 3 essential shortcuts you should know for the new Surface Pen '' . Windows Central . Jump up ^
Coppock , Mark ( 16 December 2015 ) .
</item 3>

<item 4>
Surface Pro 4
Although Surface Pro 4 is backward compatible with Type Cover accessory of Surface Pro 3 , an attachable keyboard which doubles as a screen protector , the
Surface Pro 4 Type Cover has been announced with and without a fingerprint ID sensor . The new Type Cover is thinner and lighter than its predecessor , has
improved magnetic stability for lap use , a mechanical keyboard with improved key spacing , as well as a larger touchpad . An updated version of Surface Pen is
included with the Surface Pro 4 , which has 1024 levels of pressure sensitivity and is heavier with a rubberized tip . However , the new pen has only one button
on the side rather than the previous pen 's two . There is a eraser on the end that doubles as a button , which may perform different tasks , like opening the
OneNote app or activating Cortana , and can be customized using the Surface App . Issues ( edit ) The Surface Pro 4 is known to have wide spread flicker issues
with the display named as Flickergate after the independent website flickergate.com which was sat up by the affected users after getting no fix from Microsoft .
</item 4>

<item 5>
Surface Pro 4
TechRadar.com . Retrieved 17 November 2015 . Preceded by Surface Pro 3 Surface Pro 4 4th generation Succeeded by Surface Pro ( 2017 ) Windows - powered tablet
computers and 2 - in - 1 PCs Windows 10 available Acer Iconia Tab W500 Iconia Tab W510 Iconia W700 Asus Transformer VivoTab Dell XPS 12 HP Envy X2 Lenovo
IdeaPad Yoga 11S Yoga 13 Lynx ThinkPad ThinkPad Helix ThinkPad Tablet 2 ThinkPad 8 ThinkPad 10 ThinkPad Yoga Microsoft Surface Pro Surface Pro 2 Surface Pro 3
Surface 3 Surface Pro 4 Surface Book Surface Pro ( 5th generation ) Surface Book 2 Samsung Ativ Ativ Q Galaxy TabPro S Galaxy Book Huawei MateBook Windows RT
Asus VivoTab RT Dell XPS 10 Lenovo IdeaPad Yoga 11 Nokia Lumia 2520 Surface Surface 2 Samsung Ativ Tab Microsoft History Outline People Founders Bill Gates Paul
Allen Board of directors John W.
</item 5>

<item 6>
Surface Pro 4
( hide ) Surface Pro 4 Configuration Options Price Tier ( USD ) CPU Integrated GPU RAM Internal Storage 899 Intel Core m3 - 6Y30 ( 0.9 to 2.2 GHz ) HD 515 4 GB
128 GB 999 Intel Core i5 - 6300U ( 2.4 to 3.0 GHz ) HD 520 1299 8 GB 256 GB 1499 16 GB 1699 8 GB 512 GB 1899 16 GB 1599 Intel Core i7 - 6650U ( 2.2 to 3.4 GHz )
Iris 540 8 GB 256 GB 1799 16 GB 2199 512 GB 2699 1 TB Software ( edit ) All Surface Pro 4 models come with a 64 - bit version of Windows 10 Pro and a Microsoft
Office 30 - day trial . Windows 10 comes pre-installed with Mail , Calendar , People , Xbox ( app ) , Photos , Movies and TV , Groove , and Microsoft Edge .
With Windows 10 the Tablet mode is available when the Type Cover is detached from the device .
</item 6>

<item 7>
Surface Pro 4
The common traits , inherited from the previous generation device , are a continuous kickstand , magnesium casing with a silver - colored back side , 802.11 a /
b / g / n / ac Wi - Fi radio , Bluetooth 4.0 Low Energy , full - size USB 3.0 port , microSD card reader , headset jack , Mini DisplayPort , Cover Port , and a
renamed SurfaceConnect , a port on the side to connect a wall charger , Docking Station or Surface Dock accessories .
</item 7>

<item 8>
Surface Pro 4
256 GB , 512 GB , or 1 TB Display 12.3 inches ( 31 cm ) 2736x1824 ( 267ppi ) PixelSense display with 3 : 2 aspect ratio Graphics m3 : Intel HD graphics 515 i5 :
Intel HD graphics 520 i7 : Intel Iris 540 graphics 292.1 millimetres ( 11.50 in ) ( width ) 201.42 millimetres ( 7.930 in ) ( height ) 8.45 millimetres ( 0.333
in ) ( depth ) Weight m3 : 766 grams ( 1.689 lb ) i5 : 786 grams ( 1.733 lb ) i7 : 786 grams ( 1.733 lb ) Predecessor Surface Pro 3 Successor Surface Pro ( 5th
generation ) This article is part of a series on the Microsoft Surface Surface Surface Pro Pro 2 Pro 3 Pro 4 Pro ( 2017 ) Surface Book Surface Laptop Surface
Studio Surface Hub Accessories Surface Pen Covers Dock Dial The Surface Pro 4 is the fourth - generation Surface - series 2 - in - 1 detachable , designed ,
developed ,
</item 8>

<item 9>
Surface Pro 4
With Windows 10 the Tablet mode is available when the Type Cover is detached from the device . In this mode , all windows are opened full - screen and the
interface becomes more touch - centric . The device also has a Windows Hello - compatible camera . It combines the use of a regular and infrared cameras to
authenticate the user . Accessories ( edit ) The Surface Pro 4 is backward compatible with all accessories of its predecessor . At the Windows 10 Devices Event
on 6 October , 2015 , alongside new Surface 2 - in - 1s , new accessories were announced . The Surface Dock is a redesigned docking accessory in a brick form
factor . It is compatible with all past and future Surface models with a SurfaceConnect side port , previously used to connect a wall charger or Docking Station
accessory : Surface Pro 3 , 4 and Surface Book . The Surface Dock will have 2 Mini DisplayPorts , 1 Gigabit Ethernet , 4 USB 3.0 , and 1 audio out ports .
</item 9>

<item 10>
Surface Pro 4
Microsoft 's spokesperson acknowledged the issue stating : `` We are aware that some customers have experienced a screen flicker on Surface Pro 4 and are
monitoring the situation closely . Customers impacted by this should contact Microsoft support . '' Microsoft has stated that bad batteries , pens and Type
Covers may be repaired under warranty . Reception ( edit ) The Surface Pro 4 received generally positive reviews from technology critics . Most praise a sharper
and larger display compared to the one found in the predecessor , redesigned cooling system , improved technical specifications , and an improved Type Cover .
The common disadvantages found by reviewers were that the Type Cover is still a separate purchase , the relatively modest performance of the Intel Core m3 chip
, and modest battery life . Timeline ( edit ) Source : Microsoft Devices Blog References ( edit ) Jump up ^ `` Microsoft Surface Pro 4 announced with new
Surface Pen , starts at $899 '' . The Verge . Jump up ^ `` Here are the details of the Surface Book and Surface Pro 4 chips , and why they matter '' . Jump up ^
`` ARK Intel Core m3 Specifications '' . Jump up ^ `` ARK Intel Core i5 Specifications '' .
</item 10>
=== END ITEM id=3040413436f05365 ===

=== ITEM id=7cfa29d2bd815178 ===
Gold sentences and facts:
Sentence 0: State Pre-Prohibition ( prior to 1919 ) Post-Prohibition ( after 1933 ) 1970s / 26th Amendment ( adopted in 1971 ) 1980s / Drinking Age Act of 1984 21st century Alabama Pre 1881 : None 1881 : 21 ( none with consent of parent ) 21 1975 : Lowered to 19 1986 : Raised to 21 21 ( no one underage is allowed consumption Section 28 - 1 ) Alaska ? 21 1970 : Lowered to 19 1984 : Raised to 21 with grandfather clause ( if born before January 1 , 1965 ) 21 Arizona ? 21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and liquor 18 Raised to 21 in 1987 21 Connecticut ? 21 1972 : Lowered to 18 1982 : Raised to 19 1983 : Raised to 20 1985 : Raised to 21 21 ( exemptions : ( 1 ) a person over age eighteen who is an employee or permit holder under section 30 - 90a and who possesses alcoholic liquor in the course of such person 's employment or business , ( 2 ) a minor who possesses alcoholic liquor on the order of a practicing physician , or ( 3 ) a minor who possesses alcoholic liquor while accompanied by a parent , guardian or spouse of the minor , who has attained the age of twenty - one . Nothing in this subsection shall be construed to burden a person 's exercise of religion under section 3 of article first of the Constitution of the state in violation of subsection ( a ) of section 52 - 571b . Delaware ? 21 1972 : Lowered to 20 1984 : Raised to 21 21 District of Columbia ? 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 : Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 : Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to 19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ? Post-1934 : 21 21 21 21 Iowa ? 21 1972 : Lowered to 19 in 1972 ( Jul 1 ) 1973 : Lowered to 18 in 1973 ( Jul 1 ) 1978 : Raised to 19 in 1978 ( Jul 1 ) 1986 : Raised to 21 in 1986 ( Jul 1 ) with grandfather clause . 21 Kansas ? 1949 : 18 : for beer and wine with less than 4 % ABV 21 : for liquor 18 : for beer and wine with less than 4 % ABV 21 : for beer and wine with 4 % ABV or more 21 : for liquor 1985 : Raised to 21 21 ( However , anyone under 21 may consume cereal malt beverages with parental supervision on their own property ) Kentucky ? 1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 : De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision , raising to 21 three months later . Other exceptions still remain . 21 ( Exceptions to state law include : For an established religious purpose ; When a person under twenty - one years of age is accompanied by a parent , spouse , or legal guardian twenty - one years of age or older ; For medical purposes when purchased as an over the counter medication , or when prescribed or administered by a licensed physician , pharmacist , dentist , nurse , hospital , or medical institution ; In a private residence , which shall include a residential dwelling and up to twenty contiguous acres , on which the dwelling is located , owned by the same person who owns the dwelling ; The sale , handling , transport , or service in dispensing of any alcoholic beverage pursuant to lawful ownership of an establishment or to lawful employment of a person under twenty - one years of age by a duly licensed manufacturer , wholesaler , or retailer of beverage alcohol . ) Maine ? 21 1969 : Lowered to 20 1972 : Lowered to 18 1977 : Raised to 20 1985 : Raised to 21 21 Maryland ? 21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses , teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18 in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21 Mississippi ? 1966 : 18 : for beer and wine 21 : for liquor ( alcohol not legalized until 1966 ) 18 Raised to 21 in 1986 21 Missouri ? 1945 : 21 21 21 21 Montana ? 21 1971 : Lowered to 19 1973 : Lowered to 18 1979 : Raised to 19 1987 : Raised to 21 21 Nebraska ? 21 1969 : Lowered to 20 1972 : Lowered to 19 1980 : Raised to 20 1985 : Raised to 21 21 Nevada ? 1933 : 21 21 21 21 New Hampshire ? 21 1973 : Lowered to 18 1979 : Raised to 20 1985 : Raised to 21 21 New Jersey Before 1880 : None Post-1880 : 18 ( penalties only against businesses ) 21 1973 : Lowered to 18 1980 : Raised to 19 1983 : Raised to 21 21 New Mexico ? 1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 : for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ? 1935 : 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : beer 21 : wine and liquor 1988 : 21 21 ( However , anyone can drink with parents or legal - age spouse ) Oklahoma ? Men : 21 Women : 18 : 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1976 : Lowered to 18 ( for 3.2 % ABW beer ) for both sexes in 1976 ( Dec ) Last state to lower the drinking age . 1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to 20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 21 South Dakota ? 1934 : 21 ( all ) 1939 : Lowered to 18 : for 3.2 % beer Raised to 19 : for 3.2 % beer 1972 : 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1988 : Raised to 21 21 Tennessee Pre 1881 : None 1881 : 21 21 1964 : Lowered to 18 1979 : Raised to 19 1984 : Raised to 21 21 Texas Pre 1909 : 21 None ( with written permission of parent or guardian ) Post 1909 : 21 21 1973 : Lowered to 18 1981 : Raised to 19 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 Utah ? 1935 : 21 21 21 21 Vermont ? 21 1971 : Lowered to 18 1986 : Raised to 21 21 Virginia ? 21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer ) 1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all ) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all ) 18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales , remained 18 for on - premises ) . 1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 ( however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010
  Fact 0.0: The table lists each state's drinking age in five periods: pre-Prohibition (prior to 1919), post-Prohibition (after 1933), the 1970s (26th Amendment, adopted in 1971), the 1980s (Drinking Age Act of 1984), and the 21st century.
  Fact 0.1: Alabama had no drinking age before 1881.
  Fact 0.2: Alabama set its drinking age at 21 in 1881, with none with consent of parent.
  Fact 0.3: Alabama's drinking age was 21 after Prohibition.
  Fact 0.4: Alabama lowered its drinking age to 19 in 1975.
  Fact 0.5: Alabama raised its drinking age to 21 in 1986.
  Fact 0.6: Alabama's 21st-century drinking age is 21, and no one underage is allowed consumption under Section 28-1.
  Fact 0.7: Alaska's post-Prohibition drinking age was 21.
  Fact 0.8: Alaska lowered its drinking age to 19 in 1970.
  Fact 0.9: Alaska raised its drinking age to 21 in 1984, with a grandfather clause for those born before January 1, 1965.
  Fact 0.10: Alaska's 21st-century drinking age is 21.
  Fact 0.11: Arizona's post-Prohibition drinking age was 21.
  Fact 0.12: Arizona lowered its drinking age to 19 in 1972.
  Fact 0.13: Arizona raised its drinking age to 21 in 1985.
  Fact 0.14: Arizona's 21st-century drinking age is 21.
  Fact 0.15: Arkansas had no drinking age before 1925.
  Fact 0.16: Arkansas set its drinking age at 21 in 1925.
  Fact 0.17: Arkansas's drinking age stayed at 21 in the post-Prohibition, 1970s, 1980s and 21st-century periods.
  Fact 0.18: California was regulated by municipality or county before 1891, when the common age was 16.
  Fact 0.19: California set a statewide drinking age of 18 in 1891.
  Fact 0.20: California set its drinking age at 21 in 1933.
  Fact 0.21: California's drinking age stayed at 21 after 1933, except small amounts for religious ceremonies.
  Fact 0.22: Colorado had no drinking age before Prohibition.
  Fact 0.23: Colorado set its drinking age in 1945 at 18 for beer and 21 for wine and liquor.
  Fact 0.24: Colorado's drinking age was 18 in the 1970s.
  Fact 0.25: Colorado raised its drinking age to 21 in 1987.
  Fact 0.26: Colorado's 21st-century drinking age is 21.
  Fact 0.27: Connecticut's post-Prohibition drinking age was 21.
  Fact 0.28: Connecticut lowered its drinking age to 18 in 1972.
  Fact 0.29: Connecticut raised its drinking age to 19 in 1982.
  Fact 0.30: Connecticut raised its drinking age to 20 in 1983.
  Fact 0.31: Connecticut raised its drinking age to 21 in 1985.
  Fact 0.32: Connecticut's 21st-century drinking age is 21.
  Fact 0.33: Connecticut exempts a person over 18 who is an employee or permit holder under section 30-90a and who possesses alcoholic liquor in the course of such employment or business.
  Fact 0.34: Connecticut exempts a minor who possesses alcoholic liquor on the order of a practicing physician.
  Fact 0.35: Connecticut exempts a minor who possesses alcoholic liquor while accompanied by a parent, guardian or spouse of the minor who has attained the age of 21.
  Fact 0.36: Connecticut's age limit does not burden a person's exercise of religion under section 3 of article first of the state Constitution in violation of section 52-571b(a).
  Fact 0.37: Delaware's post-Prohibition drinking age was 21.
  Fact 0.38: Delaware lowered its drinking age to 20 in 1972.
  Fact 0.39: Delaware raised its drinking age to 21 in 1984.
  Fact 0.40: Delaware's 21st-century drinking age is 21.
  Fact 0.41: The District of Columbia set its drinking age at 18 for beer and wine and 21 for liquor.
  Fact 0.42: The District of Columbia's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.43: The District of Columbia raised its drinking age to 21 in 1986, with a grandfather clause.
  Fact 0.44: The District of Columbia's 21st-century drinking age is 21.
  Fact 0.45: Florida's drinking age was 18 in the 1970s.
  Fact 0.46: Florida raised its drinking age to 19 in 1980.
  Fact 0.47: Florida raised its drinking age to 21 in 1985, with a grandfather clause.
  Fact 0.48: Florida's 21st-century drinking age is 21.
  Fact 0.49: Georgia's post-Prohibition drinking age was 21.
  Fact 0.50: Georgia lowered its drinking age to 18 in 1972.
  Fact 0.51: Georgia raised its drinking age to 19 in 1984.
  Fact 0.52: Georgia raised its drinking age to 20 in 1985.
  Fact 0.53: Georgia raised its drinking age to 21 in 1986.
  Fact 0.54: Georgia's 21st-century drinking age is 21.
  Fact 0.55: Hawaii's post-Prohibition drinking age was 21.
  Fact 0.56: Hawaii lowered its drinking age to 18 in 1972.
  Fact 0.57: Hawaii raised its drinking age to 21 in 1986.
  Fact 0.58: Hawaii's 21st-century drinking age is 21.
  Fact 0.59: Idaho's post-Prohibition drinking age was 20 for beer and 21 for wine and liquor.
  Fact 0.60: Idaho lowered its drinking age to 19 for all alcohol in 1972.
  Fact 0.61: Idaho raised its drinking age to 21 on April 11, 1987, with a grandfather clause.
  Fact 0.62: Idaho's 21st-century drinking age is 21.
  Fact 0.63: Illinois had a drinking age equal to the age of majority in 1872, with none with written consent of parent.
  Fact 0.64: Illinois's drinking age was the age of majority before 1961.
  Fact 0.65: Illinois raised its drinking age to 21 in 1961.
  Fact 0.66: Illinois lowered its drinking age to 19 for beer and wine in 1973, and kept 21 for liquor.
  Fact 0.67: Illinois raised its drinking age to 21 in 1980.
  Fact 0.68: Illinois's 21st-century drinking age is 21.
  Fact 0.69: Indiana's drinking age has been 21 since 1934.
  Fact 0.70: Indiana's drinking age was 21 in the 1970s, 1980s and 21st century.
  Fact 0.71: Iowa's post-Prohibition drinking age was 21.
  Fact 0.72: Iowa lowered its drinking age to 19 on July 1, 1972.
  Fact 0.73: Iowa lowered its drinking age to 18 on July 1, 1973.
  Fact 0.74: Iowa raised its drinking age to 19 on July 1, 1978.
  Fact 0.75: Iowa raised its drinking age to 21 on July 1, 1986, with a grandfather clause.
  Fact 0.76: Iowa's 21st-century drinking age is 21.
  Fact 0.77: Kansas set its post-Prohibition drinking age in 1949 at 18 for beer and wine under 4% ABV and 21 for liquor.
  Fact 0.78: Kansas's 1970s drinking age was 18 for beer and wine under 4% ABV, and 21 for beer and wine at 4% ABV or more and for liquor.
  Fact 0.79: Kansas raised its drinking age to 21 in 1985.
  Fact 0.80: Kansas's 21st-century drinking age is 21, but anyone under 21 may consume cereal malt beverages with parental supervision on their own property.
  Fact 0.81: Kentucky's drinking age was 21 from 1938 onward.
  Fact 0.82: Louisiana's drinking age was 18 as of 1948.
  Fact 0.83: Louisiana's drinking age was 18 in the 1970s.
  Fact 0.84: Louisiana's drinking age was raised to 21 de jure in 1987, but the de facto age stayed at 18 until 1995 because of a sale loophole.
  Fact 0.85: Louisiana's de facto drinking age was raised to 21 in 1995 when the loophole was closed.
  Fact 0.86: In 1996 the Louisiana Supreme Court briefly lowered the drinking age to 18, until it reversed its decision and raised it to 21 three months later.
  Fact 0.87: Louisiana's 21st-century drinking age is 21, with exceptions that remain.
  Fact 0.88: Louisiana's drinking age exception covers an established religious purpose.
  Fact 0.89: Louisiana's drinking age exception covers a person under 21 accompanied by a parent, spouse, or legal guardian who is 21 or older.
  Fact 0.90: Louisiana's drinking age exception covers medical purposes, when purchased as an over-the-counter medication or when prescribed or administered by a licensed physician, pharmacist, dentist, nurse, hospital, or medical institution.
  Fact 0.91: Louisiana's drinking age exception covers a private residence, including a residential dwelling and up to twenty contiguous acres on which the dwelling is located, owned by the same person who owns the dwelling.
  Fact 0.92: Louisiana's drinking age exception covers the sale, handling, transport, or service in dispensing of alcoholic beverages under lawful ownership of an establishment or lawful employment of a person under 21 by a duly licensed manufacturer, wholesaler, or retailer of beverage alcohol.
  Fact 0.93: Maine's post-Prohibition drinking age was 21.
  Fact 0.94: Maine lowered its drinking age to 20 in 1969.
  Fact 0.95: Maine lowered its drinking age to 18 in 1972.
  Fact 0.96: Maine raised its drinking age to 20 in 1977.
  Fact 0.97: Maine raised its drinking age to 21 in 1985.
  Fact 0.98: Maine's 21st-century drinking age is 21.
  Fact 0.99: Maryland's post-Prohibition drinking age was 21.
  Fact 0.100: Maryland lowered its drinking age to 18 for beer and wine in 1974, and kept 21 for liquor.
  Fact 0.101: Maryland raised its drinking age to 21 in 1982, with a grandfather clause.
  Fact 0.102: Maryland's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.103: Massachusetts's post-Prohibition drinking age was 21.
  Fact 0.104: Massachusetts lowered its drinking age to 18 in 1973.
  Fact 0.105: Massachusetts raised its drinking age to 20 on April 16, 1979.
  Fact 0.106: Massachusetts raised its drinking age to 21 in 1985.
  Fact 0.107: Massachusetts's 21st-century drinking age is 21.
  Fact 0.108: Michigan's post-Prohibition drinking age was 21.
  Fact 0.109: Michigan lowered its drinking age to 18 on January 1, 1972.
  Fact 0.110: Michigan raised its drinking age to 19 on December 3, 1978.
  Fact 0.111: Michigan raised its drinking age to 21 on December 21, 1978, 18 days later.
  Fact 0.112: Michigan was the first state to raise its age to 21 after lowering it.
  Fact 0.113: Michigan's 21st-century drinking age is 21.
  Fact 0.114: Minnesota's post-Prohibition drinking age was 21.
  Fact 0.115: Minnesota lowered its drinking age to 18 on June 1, 1973.
  Fact 0.116: Minnesota raised its drinking age to 19 on September 1, 1976.
  Fact 0.117: Minnesota was the first state to raise its drinking age after lowering it.
  Fact 0.118: Minnesota raised its drinking age to 21 in 1986.
  Fact 0.119: Minnesota's 21st-century drinking age is 21.
  Fact 0.120: Mississippi set a drinking age of 18 for beer and wine and 21 for liquor in 1966, when alcohol was legalized.
  Fact 0.121: Mississippi's drinking age was 18 in the 1970s.
  Fact 0.122: Mississippi raised its drinking age to 21 in 1986.
  Fact 0.123: Mississippi's 21st-century drinking age is 21.
  Fact 0.124: Missouri's drinking age was 21 from 1945 onward.
  Fact 0.125: Montana's post-Prohibition drinking age was 21.
  Fact 0.126: Montana lowered its drinking age to 19 in 1971.
  Fact 0.127: Montana lowered its drinking age to 18 in 1973.
  Fact 0.128: Montana raised its drinking age to 19 in 1979.
  Fact 0.129: Montana raised its drinking age to 21 in 1987.
  Fact 0.130: Montana's 21st-century drinking age is 21.
  Fact 0.131: Nebraska's post-Prohibition drinking age was 21.
  Fact 0.132: Nebraska lowered its drinking age to 20 in 1969.
  Fact 0.133: Nebraska lowered its drinking age to 19 in 1972.
  Fact 0.134: Nebraska raised its drinking age to 20 in 1980.
  Fact 0.135: Nebraska raised its drinking age to 21 in 1985.
  Fact 0.136: Nebraska's 21st-century drinking age is 21.
  Fact 0.137: Nevada's drinking age was 21 from 1933 onward.
  Fact 0.138: New Hampshire's post-Prohibition drinking age was 21.
  Fact 0.139: New Hampshire lowered its drinking age to 18 in 1973.
  Fact 0.140: New Hampshire raised its drinking age to 20 in 1979.
  Fact 0.141: New Hampshire raised its drinking age to 21 in 1985.
  Fact 0.142: New Hampshire's 21st-century drinking age is 21.
  Fact 0.143: New Jersey had no drinking age before 1880.
  Fact 0.144: New Jersey set a drinking age of 18 after 1880, with penalties only against businesses.
  Fact 0.145: New Jersey's post-Prohibition drinking age was 21.
  Fact 0.146: New Jersey lowered its drinking age to 18 in 1973.
  Fact 0.147: New Jersey raised its drinking age to 19 in 1980.
  Fact 0.148: New Jersey raised its drinking age to 21 in 1983.
  Fact 0.149: New Jersey's 21st-century drinking age is 21.
  Fact 0.150: New Mexico's drinking age was 21 from 1934 onward.
  Fact 0.151: New York's pre-Prohibition drinking age was 21.
  Fact 0.152: New York's post-Prohibition drinking age was 18.
  Fact 0.153: New York's drinking age was 18 in the 1970s.
  Fact 0.154: New York raised its drinking age to 19 in December 1982.
  Fact 0.155: New York raised its drinking age to 21 in December 1985.
  Fact 0.156: New York's 21st-century drinking age is 21.
  Fact 0.157: North Carolina's post-Prohibition drinking age was 18.
  Fact 0.158: North Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.159: North Carolina's 1983 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.160: North Carolina raised its drinking age to 21 on September 1, 1986.
  Fact 0.161: North Carolina's 21st-century drinking age is 21.
  Fact 0.162: North Dakota's drinking age was 21 from 1936 onward.
  Fact 0.163: Ohio's pre-Prohibition drinking age is unknown.
  Fact 0.164: Ohio set its post-Prohibition drinking age in 1935 at 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.165: Ohio's 1970s drinking age was 18 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.166: In 1982 Ohio set the age at 19 for 3.2% ABW beer and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.167: A second 1982 Ohio entry set the age at 19 for beer and 21 for wine and liquor.
  Fact 0.168: Ohio raised its drinking age to 21 in 1988.
  Fact 0.169: Ohio's 21st-century drinking age is 21, and anyone can drink with parents or a legal-age spouse.
  Fact 0.170: Oklahoma's drinking age was 21 for men and 18 for women for 3.2% ABW beer, and 21 for beer stronger than 3.2% ABW, wine and liquor.
  Fact 0.171: Oklahoma lowered its drinking age to 18 for 3.2% ABW beer for both sexes in December 1976.
  Fact 0.172: Oklahoma was the last state to lower the drinking age.
  Fact 0.173: Oklahoma raised its drinking age to 21 in 1983.
  Fact 0.174: Oklahoma's 21st-century drinking age is 21.
  Fact 0.175: Oregon's drinking age was 21 from 1933 onward.
  Fact 0.176: Pennsylvania's drinking age was 21 from 1935 onward.
  Fact 0.177: Rhode Island's post-Prohibition drinking age was 21.
  Fact 0.178: Rhode Island lowered its drinking age to 18 in 1970.
  Fact 0.179: Rhode Island raised its drinking age to 19 in 1980.
  Fact 0.180: Rhode Island raised its drinking age to 20 in 1981.
  Fact 0.181: Rhode Island raised its drinking age to 21 in 1984.
  Fact 0.182: Rhode Island's 21st-century drinking age is 21.
  Fact 0.183: South Carolina's pre-Prohibition drinking age was 21.
  Fact 0.184: South Carolina set its post-Prohibition drinking age in 1935 at 18 for beer and wine and 21 for liquor.
  Fact 0.185: South Carolina's 1970s drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.186: South Carolina's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.187: South Carolina raised its drinking age to 21 in 1986.
  Fact 0.188: South Carolina's 21st-century drinking age is 21.
  Fact 0.189: South Dakota's drinking age was 21 for all alcohol in 1934.
  Fact 0.190: South Dakota lowered its drinking age to 18 for 3.2% beer in 1939 and raised it to 19 for 3.2% beer.
  Fact 0.191: South Dakota's 1972 drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.192: South Dakota's 1984 drinking age was 19 for beer and wine and 21 for liquor.
  Fact 0.193: South Dakota raised its drinking age to 21 in 1988.
  Fact 0.194: South Dakota's 21st-century drinking age is 21.
  Fact 0.195: Tennessee had no drinking age before 1881.
  Fact 0.196: Tennessee set its drinking age at 21 in 1881.
  Fact 0.197: Tennessee's post-Prohibition drinking age was 21.
  Fact 0.198: Tennessee lowered its drinking age to 18 in 1964.
  Fact 0.199: Tennessee raised its drinking age to 19 in 1979.
  Fact 0.200: Tennessee raised its drinking age to 21 in 1984.
  Fact 0.201: Tennessee's 21st-century drinking age is 21.
  Fact 0.202: Texas's drinking age was 21 before 1909, with none with written permission of parent or guardian.
  Fact 0.203: Texas's drinking age was 21 after 1909 and after Prohibition.
  Fact 0.204: Texas lowered its drinking age to 18 in 1973.
  Fact 0.205: Texas raised its drinking age to 19 in 1981.
  Fact 0.206: Texas raised its drinking age to 21 on September 1, 1986.
  Fact 0.207: Texas's 21st-century drinking age is 21.
  Fact 0.208: Utah's drinking age was 21 from 1935 onward.
  Fact 0.209: Vermont's post-Prohibition drinking age was 21.
  Fact 0.210: Vermont lowered its drinking age to 18 in 1971.
  Fact 0.211: Vermont raised its drinking age to 21 in 1986.
  Fact 0.212: Vermont's 21st-century drinking age is 21.
  Fact 0.213: Virginia's post-Prohibition drinking age was 21.
  Fact 0.214: Virginia lowered its drinking age to 18 for beer and wine in 1974, while 21 for liquor remained.
  Fact 0.215: Virginia raised its drinking age to 19 for off-premises consumption in 1981.
  Fact 0.216: Virginia raised its drinking age to 19 for all beer in 1983.
  Fact 0.217: Virginia raised its drinking age to 21 in 1985.
  Fact 0.218: Virginia's 21st-century drinking age is 21.
  Fact 0.219: Washington's drinking age has been the age of majority since 1877.
  Fact 0.220: Washington's drinking age was 21 from after Prohibition onward.
  Fact 0.221: West Virginia's post-Prohibition drinking age was 18 for beer and wine and 21 for liquor.
  Fact 0.222: West Virginia lowered its drinking age to 18 for all alcohol in 1972.
  Fact 0.223: West Virginia raised its drinking age to 19 in 1983, and to 21 for non-residents.
  Fact 0.224: West Virginia raised its drinking age to 21 in 1986.
  Fact 0.225: West Virginia's 21st-century drinking age is 21.
  Fact 0.226: Wisconsin's drinking age was 18 for wine and liquor, with none for beer, from 1839 to 1866.
  Fact 0.227: Wisconsin set its drinking age at 21 for all alcohol after 1866.
  Fact 0.228: Wisconsin's post-Prohibition drinking age was 18 for beer and 21 for wine and spirits.
  Fact 0.229: In 1957 Wisconsin set a 21 limit for residents of bordering states with an age limit of 21.
  Fact 0.230: In 1963 Wisconsin raised its age to 21 for off-premises beer sales, while it remained 18 for on-premises.
  Fact 0.231: Wisconsin lowered its drinking age to 18 for all alcohol in March 1972.
  Fact 0.232: Wisconsin lifted the border state restriction in 1977.
  Fact 0.233: Wisconsin raised its drinking age to 19 in 1984.
  Fact 0.234: Wisconsin raised its drinking age to 21 on September 1, 1986.
  Fact 0.235: Wisconsin's 21st-century drinking age is 21, and anyone can drink when parents, spouses or teachers are present.
  Fact 0.236: Wyoming's post-Prohibition drinking age was 21.
  Fact 0.237: Wyoming lowered its drinking age to 19 in 1973.
  Fact 0.238: Wyoming raised its drinking age to 21 on July 1, 1988.
  Fact 0.239: Wyoming was the last state to raise its de jure drinking age to 21.
  Fact 0.240: Wyoming's 21st-century drinking age is 21.
  Fact 0.241: American Samoa's listed drinking age is 21.
  Fact 0.242: The Northern Mariana Islands' listed drinking age is 21.
  Fact 0.243: Puerto Rico's post-Prohibition drinking age was 18.
  Fact 0.244: Puerto Rico's 1970s drinking age was 18.
  Fact 0.245: Puerto Rico kept its drinking age at 18 despite a 10% highway funding penalty under the Drinking Age Act.
  Fact 0.246: Puerto Rico's 21st-century drinking age is 18.
  Fact 0.247: The Virgin Islands' drinking age was 18 in the 1980s and in the 21st century.
  Fact 0.248: Guam's listed drinking age includes 21 and 18.
  Fact 0.249: Guam's 1980s and 21st-century drinking age was 18.
  Fact 0.250: Guam raised its drinking age to 21 in 2010.
Sentence 1: From 1976 to 1983 , several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 % of their federal highway funds . By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
  Fact 1.0: From 1976 to 1983, several states voluntarily raised their purchase ages to 19 (or, less commonly, 20 or 21).
  Fact 1.1: The voluntary purchase-age increases from 1976 to 1983 were in part to combat drunk driving fatalities.
  Fact 1.2: In 1984 Congress passed the National Minimum Drinking Age Act.
  Fact 1.3: The National Minimum Drinking Age Act required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10% of their federal highway funds.
  Fact 1.4: By mid-1988, all 50 states and the District of Columbia had raised their purchase ages to 21.
  Fact 1.5: Puerto Rico, Guam, and the Virgin Islands had not raised their purchase ages to 21 by mid-1988.
  Fact 1.6: South Dakota and Wyoming were the final two states to comply with the age 21 mandate.
  Fact 1.7: The current drinking age of 21 remains a point of contention among many Americans.
  Fact 1.8: The current drinking age of 21 is higher than the age of majority, which is 18 in most states.
  Fact 1.9: The current drinking age of 21 is higher than the drinking ages of most other countries.
  Fact 1.10: The National Minimum Drinking Age Act is seen as a congressional sidestep of the tenth amendment.
  Fact 1.11: Debates over the drinking age have not been highly publicized.
  Fact 1.12: A few states have proposed legislation to lower their drinking age.
  Fact 1.13: Guam raised its drinking age to 21 in July 2010.

Retrieved items:
<item 1>
U.S. history of alcohol minimum purchase age by state
Minimum legal drinking age as of 1983 ( one year before the Drinking Age Act of 1984 was passed ) : Minimum age is 21 Minimum age is 20 Minimum age is 19 and 21
Minimum age is 19 Minimum age is 18 and 21 Minimum age is 18 The alcohol laws of the United States regarding minimum age for purchase have changed over time .
The history is given in the table below . Unless otherwise noted , if different alcohol categories have different minimum purchase ages , the age listed below
is set at the lowest age given ( e.g. if the purchase age is 18 for beer and 21 for wine or spirits , as was the case in several states , the age in the table
will read as `` 18 '' , not `` 21 '' ) . In addition , the purchase age is not necessarily the same as the minimum age for consumption of alcoholic beverages ,
although they have often been the same . As one can see in the table below , there has been much volatility in the states ' drinking ages since the repeal of
Prohibition in 1933 .
</item 1>

<item 2>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer )
1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all
) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all )
18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales ,
remained 18 for on - premises ) .
</item 2>

<item 3>
U.S. history of alcohol minimum purchase age by state
1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 :
De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision ,
raising to 21 three months later . Other exceptions still remain .
</item 3>

<item 4>
U.S. history of alcohol minimum purchase age by state
U.S. history of alcohol minimum purchase age by State - wikipedia U.S. history of alcohol minimum purchase age by State Jump to : navigation , search Minimum
legal drinking age as of 1969 ( two years before the 26th Amendment was passed ) : Minimum age is 21 Minimum age is 20 Minimum age is 19 and 21 Minimum age is
19 Minimum age is 18 and 21 Minimum age is 18 Minimum legal drinking age as of 1975 ( when almost all states had their lowest age limit ) : Minimum age is 21
Minimum age is 20 Minimum age is 19 and 21 ; see below Minimum age is 19 Minimum age is 18 and 21 ; see below Minimum age is 18 Detail on dual age limits Both
age limits apply for following states : Washington , D.C. : The legal drinking age is 18 for beer and wine , and 21 for liquor . Illinois : The legal drinking
age is 19 for beer and wine , and 21 for liquor .
</item 4>

<item 5>
U.S. history of alcohol minimum purchase age by state
Shortly after the ratification of the 21st amendment in December , most states set their purchase ages at 21 since that was the Voting age at the time . Most of
these limits remained constant until the early 1970s . From 1969 to 1976 , some 30 states lowered their purchase ages , generally to 18 . This was primarily
because the voting age was lowered from 21 to 18 in 1971 with the 26th amendment . A lot of states started to lower their minimum drinking age in response ,
most of this occurring in 1972 or 1973 . Twelve states kept their purchase ages at 21 since repeal of Prohibition and never changed them . From 1976 to 1983 ,
several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress
passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 %
of their federal highway funds .
</item 5>

<item 6>
U.S. history of alcohol minimum purchase age by state
1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 (
however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to
raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway
funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010 Additional Notes ( edit ) Contrary to popular
belief , since the National Minimum Drinking Age Act of 1984 , not all states specifically prohibit minors ' and young adults ' consumption of alcohol in
private settings .
</item 6>

<item 7>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses ,
teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18
in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was
lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21
Mississippi ?
</item 7>

<item 8>
U.S. history of alcohol minimum purchase age by state
21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common
age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and
liquor 18 Raised to 21 in 1987 21 Connecticut ?
</item 8>

<item 9>
U.S. history of alcohol minimum purchase age by state
By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see
Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point
of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other
countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly
publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
</item 9>

<item 10>
U.S. history of alcohol minimum purchase age by state
1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 :
for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ?
</item 10>
=== END ITEM id=7cfa29d2bd815178 ===
