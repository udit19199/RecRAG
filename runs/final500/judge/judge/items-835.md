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

=== ITEM id=0d2f1efd42702d7f ===
Gold sentences and facts:
Sentence 0: At the Council , following advice offered by Simon Peter ( Acts 15 : 7 -- 11 and Acts 15 : 14 ) , Barnabas and Paul gave an account of their ministry among the gentiles ( Acts 15 : 12 ) , and the apostle James quoted from the words of the prophet Amos ( Acts 15 : 16 -- 17 , quoting Amos 9 : 11 -- 12 ) . James added his own words to the quotation : `` Known to God from eternity are all His works '' and then submitted a proposal , which was accepted by the Church and became known as the Apostolic Decree :
  Fact 0.0: Simon Peter offered advice at the Council, recorded in Acts 15:7-11 and Acts 15:14.
  Fact 0.1: Barnabas and Paul gave an account of their ministry among the gentiles at the Council, recorded in Acts 15:12.
  Fact 0.2: The account given by Barnabas and Paul followed advice offered by Simon Peter.
  Fact 0.3: The apostle James quoted from the words of the prophet Amos, recorded in Acts 15:16-17.
  Fact 0.4: The apostle James quoted Amos 9:11-12.
  Fact 0.5: James added his own words to the quotation: 'Known to God from eternity are all His works'.
  Fact 0.6: James submitted a proposal.
  Fact 0.7: The proposal was accepted by the Church.
  Fact 0.8: The proposal became known as the Apostolic Decree.

Retrieved items:
<item 1>
Council of Jerusalem
, Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early Roman Period '' , volume 3 , p. 744 ( 2008 ) .
Cambridge University Press . Jump up ^ `` However , numerous scholars have challenged the historicity of the Jerusalem Council as related by Acts , Paul 's
presence there in the manner that Luke describes , the issue of idol - food being thrust on Paul 's Gentile mission , and the historical reliability of Acts in
general . '' , Fotopolous , `` Food Offered to Idols in Roman Corinth : a socio - rhetorical reconsideration '' , pp. 181 -- 182 ( 2003 ) . Mohr Siebeck . Jump
up ^ `` Sahlin rejects the historicity of Acts completely ( Der Messias und das Gottesvolk ( 1945 ) ) . Haenchen 's view is that the Apostolic Council `` is an
imaginary construction answering to no historical reality '' ( The Acts of the Apostles ( Engtr 1971 ) , p. 463 ) .
</item 1>

<item 2>
Council of Jerusalem
In the view of some scholars , the reminder of James and the elders here is an expression of concern that Paul was not fully teaching the decision of the
Jerusalem Council 's letter to Gentiles , particularly in regard to non-strangled kosher meat , which contrasts with Paul 's advice to Gentiles in Corinth , to
`` eat whatever is sold in the meat markets '' ( 1 Corinthians 10 : 25 ) . Historicity ( edit ) See also : Historical reliability of the Acts of the Apostles
The description of the Apostolic Council in Acts 15 , generally considered the same event described in Galatians 2 , is considered by some scholars to be
contradictory to the Galatians account . The historicity of Luke 's account has been challenged , and was rejected completely by some scholars in the mid to
late 20th century . However , more recent scholarship inclines towards treating the Jerusalem Council and its rulings as a historical event , though this is
sometimes expressed with caution .
</item 2>

<item 3>
Council of Jerusalem
70 ) . Interpreting the Council 's decision ( edit ) See also : Paul the Apostle and Judaism and Early Christianity Part of a series on Jewish Christianity
Figures Jesus John the Baptist Simon Peter Twelve Apostles James , brother of Jesus Simeon of Jerusalem Jude Paul Ancient groups Ebionites Elcesaites Nazarenes
Pejoratives Judaizers Legalists Pauline Christianity Recent groups Hebrew Christian movement Messianic Judaism Saint Thomas Christians Adversity Split of
Christianity and Judaism Paul and Judaism Marcionism Christian anti-semitism Constantine Writings Gospel of Matthew Epistle of James Clementine Didache Liturgy
of St James Book of Elchasai Jewish -- Christian gospels Gospel of the Ebionites Gospel of the Hebrews Gospel of the Nazarenes Issues Aramaic of Jesus Yeshua (
name ) Council of Jerusalem Expounding of the Law Sabbath Quartodecimanism Noahide laws James 's Apostolic Decree was that most Mosaic law , including the
requirement for circumcision of males , was not obligatory for Gentile converts , possibly in order to make it easier for them to join the movement .
</item 3>

<item 4>
Council of Jerusalem
The council decided that Gentile converts to Christianity were not obligated to keep most of the Law of Moses , including the rules concerning circumcision of
males . The Council did , however , retain the prohibitions on eating blood , meat containing blood , and meat of animals not properly slain , and on
fornication and idolatry , sometimes referred to as the Apostolic Decree or Jerusalem Quadrilateral . Accounts of the council are found in Acts of the Apostles
chapter 15 ( in two different forms , the Alexandrian and Western versions ) and also possibly in Paul 's letter to the Galatians chapter 2 . Some scholars
dispute that Galatians 2 is about the Council of Jerusalem ( notably because Galatians 2 describes a private meeting ) while other scholars dispute the
historical reliability of the Acts of the Apostles . Contents ( hide ) 1 Historical background 2 Issues and outcome 3 Historicity 4 Interpreting the Council 's
decision 4.1 Modern interpretation 4.1.
</item 4>

<item 5>
Council of Jerusalem
463 ) . Dibelius ' view ( Studies in the Acts of the Apostles ( Engtr 1956 ) , pp. 93 -- 101 ) is that Luke 's treatment is literary - theological and can make
no claim to historical worth. `` , Mounce , `` Apostolic Council '' , in Bromiley ( ed . ) `` The International Standard Bible Encyclopedia '' , volume 1 , p.
200 ( rev . ed. 2001 ) . Wm . B. Eerdmans . Jump up ^ `` There is an increasing trend among scholars toward considering the Jerusalem Council as historical
event . An overwhelming majority identifies the reference to the Jerusalem Council in Acts 15 with Paul 's account in Gal. 2.1 -- 10 , and this accord is not
just limited to the historicity of the gathering alone but extends also to the authenticity of the arguments deriving from the Jerusalem church itself . '' ,
Philip , `` The Origins of Pauline Pneumatology : the Eschatological Bestowal of the Spirit '' , Wissenschaftliche Untersuchungen zum Neuen Testament 2 , Reihe
, p. 205 ( 2005 ) .
</item 5>

<item 6>
Council of Jerusalem
Authorship and development ( show ) Authorship Dating Hebrew canon Old Testament canon New Testament canon Documentary hypothesis Mosaic authorship Pauline
epistles Petrine epistles Johannine works Translations and manuscripts ( show ) Samaritan Torah Dead Sea scrolls Masoretic Text Targumim Peshitta Septuagint
Vulgate Gothic Bible Vetus Latina Luther Bible English Bibles Biblical studies ( show ) Archeology Artifacts Dating Historicity Internal consistency People
Places Names Novum Testamentum Graece Documentary hypothesis Synoptic problem NT textual categories Biblical criticism Historical Textual Source Form Redaction
Canonical Interpretation ( show ) Hermeneutics Pesher Midrash Pardes Allegorical interpretation Literalism Prophecy Inspiration Perspectives ( show ) Gnostic
Islamic Qur'anic Muhammad Inerrancy Infallibility Criticism of the Bible Outline of Bible - related topics Bible book Bible portal The Council of Jerusalem or
Apostolic Council was held in Jerusalem around AD 50 . It is unique among the ancient pre-ecumenical councils in that it is considered by Catholics and Orthodox
to be a prototype and forerunner of the later ecumenical councils and a key part of Christian ethics .
</item 6>

<item 7>
Council of Jerusalem
At the Council , following advice offered by Simon Peter ( Acts 15 : 7 -- 11 and Acts 15 : 14 ) , Barnabas and Paul gave an account of their ministry among the
gentiles ( Acts 15 : 12 ) , and the apostle James quoted from the words of the prophet Amos ( Acts 15 : 16 -- 17 , quoting Amos 9 : 11 -- 12 ) . James added his
own words to the quotation : `` Known to God from eternity are all His works '' and then submitted a proposal , which was accepted by the Church and became
known as the Apostolic Decree : It is my judgment , therefore , that we should not make it difficult for the Gentiles who are turning to God . Instead we should
write to them , telling them to abstain from food polluted by idols , from sexual immorality , from the meat of strangled animals and from blood . For the law
of Moses has been preached in every city from the earliest times and is read in the synagogues on every Sabbath .
</item 7>

<item 8>
Graph query row:
apostle: Simon Peter
council: Council of Jerusalem
relationship: PARTICIPATED_IN
connection: None
</item 8>

<item 9>
Graph query row:
apostle: Simon Peter
council: Council of Jerusalem
relationship: RELATED_TO
connection: None
</item 9>

<item 10>
Council of Jerusalem
, Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early Roman Period '' , volume 3 , p. 744 ( 2008 ) .
Cambridge University Press . Jump up ^ `` Paul 's account of the Jerusalem Council in Galatians 2 and the account of it recorded in Acts have been considered by
some scholars as being in open contradiction . '' , Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early
Roman Period '' , volume 3 , p. 744 ( 2008 ) . Cambridge University Press . Jump up ^ `` There is a very strong case against the historicity of Luke 's account
of the Apostolic Council '' , Esler , `` Community and Gospel in Luke - Acts : The Social and Political Motivations of Lucan Theology '' , p. 97 ( 1989 ) .
Cambridge University Press . Jump up ^ `` The historicity of Luke 's account in Acts 15 has been questioned on a number of grounds . ''
</item 10>
=== END ITEM id=0d2f1efd42702d7f ===

=== ITEM id=1278108785920fce ===
Gold sentences and facts:
Sentence 0: Originally introduced as a classical tax system , in which companies were subject to tax on their profits and companies ' shareholders were also liable to income tax on the dividends that they received , the first major amendment to corporation tax saw it move to a dividend imputation system in 1973 , under which an individual receiving a dividend became entitled to an income tax credit representing the corporation tax already paid by the company paying the dividend . The classical system was reintroduced in 1999 , with the abolition of advance corporation tax and of repayable dividend tax credits . Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 - 2010 to a flat rate of 20 % as of April 2015 .
  Fact 0.0: Corporation tax was originally introduced as a classical tax system.
  Fact 0.1: Under the classical system, companies were subject to tax on their profits.
  Fact 0.2: Under the classical system, companies' shareholders were also liable to income tax on the dividends they received.
  Fact 0.3: The first major amendment to corporation tax moved it to a dividend imputation system in 1973.
  Fact 0.4: Under the 1973 dividend imputation system, an individual receiving a dividend became entitled to an income tax credit representing the corporation tax already paid by the company paying the dividend.
  Fact 0.5: The classical system was reintroduced in 1999.
  Fact 0.6: The 1999 reintroduction of the classical system came with the abolition of advance corporation tax.
  Fact 0.7: The 1999 reintroduction of the classical system came with the abolition of repayable dividend tax credits.
  Fact 0.8: Another change saw the single main rate of tax split into three.
  Fact 0.9: Tax competition between jurisdictions reduced the main corporate tax rate from 28% in 2008-2010.
  Fact 0.10: Tax competition between jurisdictions reduced the main corporate tax rate to a flat rate of 20% as of April 2015.

Retrieved items:
<item 1>
United Kingdom corporation tax
There was a matching reduction in the basic income tax rate on dividends to 10 % , while a new higher - rate of 32.5 % was introduced which led to an overall
effective 25 % tax rate for higher rate taxpayers on dividends ( after setting this `` notional '' tax credit against the tax liability ). While non-taxpayers
were no longer able to claim this amount from the treasury ( as opposed to taxpayers who could deduct it from their tax bill ) , the 20 % ACT ( which would have
previously been deducted from the dividend before payment ) was no longer levied . The abolition of ACT was actually one of the biggest tax grabs ever by a
Chancellor , as pension companies could no longer recover the tax paid on dividends ( along with pensioners and other reliant on the income ) . This now costs
over £ 10bn a year and was one of the factors that lead to the death of the final salary pension scheme . ( ref :
http://www.telegraph.co.uk/finance/personalfinance/pensions/10798785/True-cost-of-Labours-pension-tax-raid-and-others-since-Seventies.html ) .
</item 1>

<item 2>
United Kingdom corporation tax
Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 -
2010 to a flat rate of 20 % as of April 2015 . The UK government faced problems with its corporate tax structure , including European Court of Justice
judgements that aspects of it are incompatible with EU treaties . Tax avoidance schemes marketed by the financial sector have also proven an irritant , and been
countered by complicated anti-avoidance legislation . The complexity of the corporation tax system is a recognised issue . The Labour government , supported by
the Opposition parties , carried through wide - scale reform from the Tax Law Rewrite project , resulting in the Corporation Tax Act 2010 . The tax has slowly
been integrating generally accepted accounting practice , with the corporation tax system in various specific areas based directly on the accounting treatment .
Total net corporation tax receipts in 2014 - 15 were £ 43.0 billion , which was an increase of 7 % from £ 40.3 billion in 2013 - 14 .
</item 2>

<item 3>
United Kingdom corporation tax
The 1979 Conservative Budget of Geoffrey Howe cut the small companies ' rate to 40 % , followed by a further cut in the 1982 Budget to 38 % . The Budgets of
1983 -- 1988 saw sharp cuts in both main and small companies ' rates , falling to 35 % and 25 % respectively . Budgets between 1988 and 2001 brought further
falls to a 30 % main rate and 19 % small companies ' rates . From April 1983 to March 1997 the small companies ' rate was pegged to the basic rate of income tax
. During the 1980s there was briefly a higher rate of tax imposed for capital profits . Chancellor Gordon Brown 's 1999 Budget introduced a 10 % starting rate
for profits from £ 0 to £ 10,000 , effective from April 2000 . Marginal relief applied meaning companies with profits of between £ 10,000 and £ 50,000 paid a
rate between the starting rate and the small companies ' rate ( 19 % in 2000 ) .
</item 3>

<item 4>
United Kingdom corporation tax
This ensured that where a company paid below the small companies ' rate ( 19 % in 2004 ) , dividend payments made to non-corporates ( for example , individuals
, trusts and personal representatives of deceased persons ) would be subject to additional corporation tax , bringing the corporation tax paid up to 19 % . For
example , a company making £ 10,000 profit , and making a £ 6,000 dividend distribution to an individual and £ 4,000 to another company would pay 19 %
corporation tax on the £ 6,000 . Although this measure substantially reduced the number of small businesses incorporating , the Chancellor in the 2006 Budget
said tax avoidance by small businesses through incorporation was still a major issue , and scrapped the starting rate entirely . Historic tax revenues ( edit )
UK Corporation Tax Revenue 1999 to 2016 10,000 20,000 30,000 40,000 50,000 Taxable profits and accounting profits ( edit ) The starting point for computing
taxable profits is profits before tax ( except for a life assurance company ) .
</item 4>

<item 5>
United Kingdom corporation tax
United Kingdom corporation tax - wikipedia United Kingdom corporation tax Jump to : navigation , search Throughout this article , the unqualified term `` pound
'' and the £ symbol refer to the Pound sterling . UK Corporation Tax Receipts from 1999 - 2016 , both in absolute terms and as % of GDP . Taxation in the United
Kingdom UK Government Departments HM Treasury HM Revenue and Customs UK Government VAT Income tax National Insurance Corporation tax Capital gains tax Motoring
taxes Inheritance tax Stamp duty Insurance Premium Tax Air Passenger Duty PAYE Scottish Government Revenue Scotland Scottish income tax Land and Buildings
Transaction Tax Scottish Landfill Tax Air Departure Tax Welsh Government Welsh Revenue Authority Land Transaction Tax Landfill Disposals Tax Local Government
Council Tax Domestic rates in Northern Ireland Business rates in England Business rates in Wales Business rates in Scotland Business rates in Northern Ireland
In the United Kingdom , corporation tax is a corporate tax levied in the United Kingdom on the profits made by UK - resident companies and on the profits of
entities registered overseas with permanent establishments in the UK . Until 1 April 1965 , companies were taxed at the same income tax rates as individual
taxpayers , with an additional profits tax levied on companies .
</item 5>

<item 6>
United Kingdom corporation tax
The 2002 Budget cut the starting rate to zero , with marginal relief applying in the same way . This caused a significant increase in the number of companies
being incorporated , as businesses that had operated as self - employed , paying income tax on profits from just over £ 5,000 , were attracted to the
corporation tax rate of 0 % on income up to £ 10,000 . Previously self - employed individuals could now distribute profits as dividend payments rather than
salaries . For companies with profits under £ 50,000 the corporation tax rate varied between 0 % and 19 % . Because dividend payments come with a basic rate tax
credit , provided the recipient did not earn more than the basic rate allowance , no further tax would be paid . The number of new companies being formed in
2002 -- 2003 reached 325,900 , an increase of 45 % on 2001 -- 2002 . The fact that individuals operating in this manner could potentially pay no tax at all was
felt by the government to be unfair tax avoidance , and the 2004 Budget introduced a Non-Corporate Distribution Rate .
</item 6>

<item 7>
United Kingdom corporation tax
^ Jump up to : `` Finance Act 1965 ( c. 25 ) , from UK Statute Law Database '' . UK Statutory Publications Office , Ministry of Justice . Retrieved 9 May 2007 .
Jump up ^ Tax Law Rewrite Archived 18 April 2006 at the Wayback Machine. , Her Majesty 's Revenue and Customs ( HMRC ) . Retrieved 17 April 2007 Jump up ^ ( 1 )
Income and Corporation Taxes Act 1988 ( c. 1 ) , Office of Public Sector Information , responsible for the operation of Her Majesty 's Stationery Office ( HMSO
) , ISBN 0 - 10 - 540188 - 9 Jump up ^ Most income tax rules have been rewritten in the Income Tax ( Earnings and Pensions ) Act 2003 , Income Tax ( Trading and
Other Income ) Act 2005 , and Income Tax Act 2007 . ^ Jump up to : Rates of Corporation Tax ( PDF ) Archived 9 June 2007 at the Wayback Machine. , HMRC .
</item 7>

<item 8>
United Kingdom corporation tax
Corporate Tax rates for Financial Years 2008 -- 2014 2008 -- 2010 2011 2012 2013 2014 Small profits rate 21 % 20 % 20 % 20 % 20 % Small profits upper limit £
300,000 £ 300,000 £ 300,000 £ 300,000 £ 300,000 Marginal relief limits £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £
1,500,000 £ 300,001 -- £ 1,500,000 Main rate 28 % 26 % 24 % 23 % 21 % Notes : The bands shown on the right hand side are divided by one plus the number of
associates ( usually the only associates a company has are fellow group companies , but the term is more widely defined ) The reduced rates do not apply to
close investment holding companies ( companies controlled by fewer than 5 people ( plus associates ) or by their directors / managers , whose main activity is
the holding of investments ) .
</item 8>

<item 9>
United Kingdom corporation tax
Relief for expenses ( edit ) Most direct expenses are deductible when calculating taxable income and chargeable gains . Notable exceptions include any costs of
entertaining clients . Companies with investment business may deduct certain indirect expenses known as `` expenses of management '' when calculating their
taxable profits . A similar relief is available for expenses of a life assurance company taxed on the I minus E basis which relate to the company 's basic life
assurance and general annuity business . Donations made to charities are also normally deducted in calculating taxable income if made under Gift Aid . Rates and
payment ( edit ) The 2007 Budget announced a main rate cut from 30 % to 28 % , effective from April 2008 . At the same time , the small companies ' rate was
increased from 19 % to 20 % from April 2007 , 21 % in April 2008 , to stop `` individuals artificially incorporating as small companies to avoid paying their
due share of tax , a practice if left unaddressed would cost the rest of the taxpaying population billions of pounds '' . The rate of corporation tax is
determined by the financial year , which runs from 1 April to the following 31 March .
</item 9>

<item 10>
United Kingdom corporation tax
, HMRC . Retrieved 13 April 2007 Jump up ^ `` Corporation Tax rates and reliefs - GOV.UK '' . www.gov.uk . Retrieved 2015 - 10 - 13 . ^ Jump up to :
Metallgesellschaft Ltd and Others , Hoechst AG , Hoechst UK Ltd and Commissioners of Inland Revenue , H.M. Attorney General ECJ Cases C - 397 / 98 and 410 / 98
( Joined cases ) , European Court of Justice , 2001 . Retrieved 9 May 2007 Jump up ^ `` Corporation Tax Statistics : Analyses of Corporation Tax receipts and
liabilities '' ( PDF ) . HM Revenue & Customs . 28 August 2015 . p. 8 . Archived from the original ( PDF ) on 4 March 2016 . ^ Jump up to : Corporate Tax ( PDF
) Jump up ^ Daunton , Martin ( 2002 ) . Just Taxes : The Politics of Taxation in Britain , 1914 -- 1979 ( pdf ) . Cambridge University Press .
</item 10>
=== END ITEM id=1278108785920fce ===

=== ITEM id=06cb55d44947d9cf ===
Gold sentences and facts:
Sentence 0: The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans .
  Fact 0.0: Trade winds are the prevailing pattern of easterly surface winds found in the tropics.
  Fact 0.1: Trade winds are found within the lower portion of the Earth's atmosphere.
  Fact 0.2: Trade winds are found in the lower section of the troposphere near the Earth's equator.
  Fact 0.3: Trade winds blow predominantly from the northeast in the Northern Hemisphere.
  Fact 0.4: Trade winds blow predominantly from the southeast in the Southern Hemisphere.
  Fact 0.5: Trade winds strengthen during the winter.
  Fact 0.6: Trade winds strengthen when the Arctic oscillation is in its warm phase.
  Fact 0.7: Trade winds have been used by captains of sailing ships to cross the world's oceans for centuries.
  Fact 0.8: Trade winds enabled European empire expansion into the Americas.
  Fact 0.9: Trade winds enabled trade routes to become established across the Atlantic and Pacific oceans.

Retrieved items:
<item 1>
Trade winds
The subsident ( sinking ) air is relatively dry because as it descends , the temperature increases , but the absolute humidity remains constant , which lowers
the relative humidity of the air mass . This warm , dry air is known as a superior air mass and normally resides above a maritime tropical ( warm and moist )
air mass . An increase of temperature with height is known as a temperature inversion . When it occurs within a trade wind regime , it is known as a trade wind
inversion . The surface air that flows from these subtropical high - pressure belts toward the Equator is deflected toward the west in both hemispheres by the
Coriolis effect . These winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere . Because winds
are named for the direction from which the wind is blowing , these winds are called the northeasterly trade winds in the Northern Hemisphere and the
southeasterly trade winds in the Southern Hemisphere . The trade winds of both hemispheres meet at the doldrums . As they blow across tropical regions , air
masses heat up over lower latitudes due to more direct sunlight .
</item 1>

<item 2>
Trade winds
Shallow cumulus clouds are seen within trade wind regimes , and are capped from becoming taller by a trade wind inversion , which is caused by descending air
aloft from within the subtropical ridge . The weaker the trade winds become , the more rainfall can be expected in the neighboring landmasses . Contents ( hide
) 1 History 2 Cause 3 Weather effects 4 See also 5 References History ( edit ) A Spanish galleon See also : Age of Discovery , Volta do Mar , and Age of sail
The term trade winds originally derives from the early fourteenth century late Middle English word ' trade , ' meaning `` path '' or `` track . '' The
Portuguese recognized the importance of the trade winds ( then the Volta do mar , meaning in Portuguese `` turn of the sea '' but also `` return from the sea ''
) in navigation in both the north and south Atlantic ocean as early as the 15th century . From West Africa , the Portuguese had to sail away from continental
Portugal , that is , to west and northwest . They could then turn northeast , to the area around the Azores islands , and finally east to mainland Europe .
</item 2>

<item 3>
Trade winds
Trade winds - wikipedia Trade winds Jump to : navigation , search This article is about the weather phenomenon . For other uses , see Tradewind . The westerlies
( blue arrows ) and trade winds ( yellow and brown arrows ) The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within
the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the
northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in
its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into
the Americas and trade routes to become established across the Atlantic and Pacific oceans . In meteorology , the trade winds act as the steering flow for
tropical storms that form over the Atlantic , Pacific , and southern Indian Oceans and make landfall in North America , Southeast Asia , and Madagascar and
eastern Africa , respectively . Trade winds also transport African dust westward across the Atlantic Ocean into the Caribbean Sea , as well as portions of
southeastern North America .
</item 3>

<item 4>
Trade winds
Between 1847 and 1849 , Matthew Fontaine Maury collected enough information to create wind and current charts for the world 's oceans . Cause ( edit ) 3D map
showing Hadley cells in relationship to trade winds on the surface . See also : Air mass , Hadley cell , Humidity , Intertropical Convergence Zone , Monsoon ,
Monsoon trough , Near - equatorial trough , and Prevailing winds As part of the Hadley cell circulation , surface air flows toward the equator while the flow
aloft is towards the poles . A low - pressure area of calm , light variable winds near the equator is known as the doldrums , near - equatorial trough ,
intertropical front , or the Intertropical Convergence Zone . When located within a monsoon region , this zone of low pressure and wind convergence is also
known as the monsoon trough . Around 30 ° in both hemispheres , air begins to descend toward the surface in subtropical high - pressure belts known as
subtropical ridges .
</item 4>

<item 5>
Trade winds
ISBN 978 - 1 - 4437 - 2300 - 8 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Doldrums '' . American Meteorological Society .
Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Intertropical Convergence Zone '' . American Meteorological Society . Retrieved
2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Monsoon Trough '' . American Meteorological Society . Archived from the original on 2009 -
06 - 17 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Superior air '' . American Meteorological Society . Retrieved 2009 -
10 - 28 . Jump up ^ Glossary of Meteorology ( 2009 ) . `` trade winds '' . Glossary of Meteorology .
</item 5>

<item 6>
Trade winds
Those that develop over land ( continental ) are drier and hotter than those that develop over oceans ( maritime ) , and travel northward on the western
periphery of the subtropical ridge . Maritime tropical air masses are sometimes referred to as trade air masses . The one region of the Earth which has an
absence of trade winds is the north Indian ocean . Weather effects ( edit ) Clouds which form above regions within trade wind regimes are typically composed of
cumulus which extend no more than 4 kilometres ( 13,000 ft ) in height , and are capped from being taller by the trade wind inversion . Trade winds originate
more from the direction of the poles ( northeast in the Northern Hemisphere , southeast in the Southern Hemisphere ) during the cold season , and are stronger
in the winter than the summer . As an example , the windy season in the Guianas , which lie at low latitudes in South America , occurs between January and April
. When the phase of the Arctic oscillation ( AO ) is warm , trade winds are stronger within the tropics . The cold phase of the AO leads to weaker trade winds .
</item 6>

<item 7>
Trade winds
`` trade winds '' . Glossary of Meteorology . American Meteorological Society . Archived from the original on 2008 - 12 - 11 . Retrieved 2008 - 09 - 08 . Jump
up ^ Ralph Stockman Tarr and Frank Morton McMurry ( 1909 ). Advanced geography . W.W. Shannon , State Printing , pp. 246 . Retrieved on 2009 - 04 - 15 . Jump up
^ JetStream ( 2008 ) . `` How to read weather maps '' . National Weather Service . Retrieved 2007 - 05 - 16 . Jump up ^ Glossary of Meteorology ( June 2000 ) .
`` Tropical air '' . American Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Trade air '' . American
Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ John E. Oliver ( 2005 ) . Encyclopedia of world climatology . Springer .
</item 7>

<item 8>
Trade winds
The cold phase of the AO leads to weaker trade winds . When the trade winds are weaker , more extensive areas of rain fall upon landmasses within the tropics ,
such as Central America . During mid-summer in the Northern Hemisphere ( July ) , the westward - moving trade winds south of the northward - moving subtropical
ridge expand northwestward from the Caribbean sea into southeastern North America ( Florida and Gulf Coast ) . When dust from the Sahara moving around the
southern periphery of the ridge travels over land , rainfall is suppressed and the sky changes from a blue to a white appearance which leads to an increase in
red sunsets . Its presence negatively impacts air quality by adding to the count of airborne particulates . Although the Southeast USA has some of the cleanest
air in North America , much of the African dust that reaches the United States affects Florida . Since 1970 , dust outbreaks have worsened due to periods of
drought in Africa . There is a large variability in the dust transport to the Caribbean and Florida from year to year . Dust events have been linked to a
decline in the health of coral reefs across the Caribbean and Florida , primarily since the 1970s .
</item 8>

<item 9>
Trade winds
They also learned that to reach South Africa , they needed to go far out in the ocean , head for Brazil , and around 30 ° S go east again . Following the
African coast southbound means upwind in the Southern hemisphere . In the Pacific ocean , the full wind circulation , which included both the trade wind
easterlies and higher - latitude Westerlies , was unknown to Europeans until Andres de Urdaneta 's voyage in 1565 . The captain of a sailing ship seeks a course
along which the winds can be expected to blow in the direction of travel . During the Age of Sail , the pattern of prevailing winds made various points of the
globe easy or difficult to access , and therefore had a direct effect on European empire - building and thus on modern political geography . For example ,
Manila galleons could not sail into the wind at all . By the 18th century the importance of the trade winds to England 's merchant fleet for crossing the
Atlantic Ocean had led both the general public and etymologists to identify the name with a later meaning of ' trade ' , `` ( foreign ) commerce '' .
</item 9>

<item 10>
Trade winds
See also ( edit ) Intertropical Convergence Zone Winds in the Age of Sail Westerly wind burst Volta do mar References ( edit ) Jump up ^ Carol G. Braham ; Enid
Pearsons ; Deborah M. Posner ; Georgia S. Maas & Richard Goodman ( 2001 ) . Random House Webster 's College Dictionary ( second ed . ) . Random House . p. 1385
. ISBN 0 - 375 - 42560 - 8 . Jump up ^ Hermann R. Muelder ( 2007 ) . Years of This Land - A Geographical History of the United States . Read Books . p. 38 .
ISBN 978 - 1 - 4067 - 7740 - 6 . Retrieved 2009 - 11 - 09 . ^ Jump up to : Derek Hayes ( 2001 ) . Historical atlas of the North Pacific Ocean : maps of
discovery and scientific exploration , 1500 - 2000 . Douglas & McIntyre . p. 18 . ISBN 978 - 1 - 55054 - 865 - 5 .
</item 10>
=== END ITEM id=06cb55d44947d9cf ===

=== ITEM id=a193bad5cd116ae7 ===
Gold sentences and facts:
Sentence 0: Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out .
  Fact 0.0: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 0.1: Jane and Rafael's plan to buy the apartment falls apart when Mateo breaks an antique $4,000 clock.
  Fact 0.2: Rafael moves into the Villanueva house.
  Fact 0.3: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 0.4: Rafael moves back out of the Villanueva house after the argument.
  Fact 0.5: Xiomara starts chemotherapy.
  Fact 0.6: Xiomara makes a friend named Donna.
  Fact 0.7: The timeline moves forward nine weeks.
  Fact 0.8: Donna dies of unforeseen consequences.
  Fact 0.9: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 0.10: Alba and Rafael are still in a fight.
  Fact 0.11: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 0.12: Alba breaks down to Jane.
  Fact 0.13: Alba and Rafael make up.
  Fact 0.14: Xiomara bonds with River over telenovelas.
  Fact 0.15: Xiomara accidentally eats a pot brownie.
  Fact 0.16: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 0.17: Jane Ramos gets herself disbarred.
  Fact 0.18: As a result of Jane Ramos being disbarred, the case against Petra is thrown out.
Sentence 1: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 65 `` Chapter Sixty - Five '' Brad Silberling Jennie Snyder Urman & Paul Sciarrotta October 13 , 2017 ( 2017 - 10 - 13 ) 0.68 Picking up immediately after where Season 3 left off , Jane is reunited with her first love , Adam , who found the love letter that Michael wrote . She is surprised , but happy , to see him . However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom . However , when Adam gets a job opportunity out of state , he asks for Jane 's opinion on if he should stay or go . Meanwhile , Rogelio continues to face issues with his co-star at work and things are about to get even worse . Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby . Also , Petra and Rafael begin working together to try and buy back the hotel from Luisa , but they are running into trouble involving Anezka , causing tension between them . 67 `` Chapter Sixty - Seven '' Fernando Sariñana Carolina Rivera & Micah Schraft October 27 , 2017 ( 2017 - 10 - 27 ) 0.60 Jane and Adam 's relationship seems to be moving on , and Jane thinks that it might be time for him to meet her son Mateo , but Rafael is not happy with this idea . Meanwhile , as the arrival of Darci 's baby approaches , Rogelio 's jealously of her new boyfriend increases as well . Trying to be supportive , Xo encourages Rogelio to step up and be the father he was meant to be . Elsewhere , Rafael is determined to get his hotel and life back from Luisa and will do whatever he has to do to make it happen , with or without Petra 's help . 68 `` Chapter Sixty - Eight '' Gina Lamar Deirdre Shaw & Chantelle M. Wells November 3 , 2017 ( 2017 - 11 - 03 ) 0.69 Jane becomes angry when she learns that Rafael introduced Mateo to his new girlfriend , Katherine , without her consent because she thinks Katherine is a negative influence on both Rafael and Mateo . Jane turns to Petra for help with the situation . Meanwhile , Rogelio and Darci struggle to choose a name for the baby . Elsewhere , Xo , who is focused on opening her own dance studio , decides to go into business with an unlikely foe . 69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not . Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy ) done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day . Meanwhile , Rafael gets a new job , but it is not exactly what he was hoping for . Also , Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa , but Rafael wants nothing to do with the scheme . 71 7 `` Chapter Seventy - One '' Micah Schraft Valentina Garza & Deidre Shaw December 8 , 2017 ( 2017 - 12 - 08 ) 0.65 Anezka is revealed to have faked her death to see where her loyalties lie ; initially , she trusts Petra after believing she saw her cry , but after finding out she was n't crying because of her fake death , she threatens Petra . Later , Anezka falls from Petra 's apartment to her death ... for real . Meanwhile , Xiomara and Jane set off on Jane 's book tour , when Alba decides to tag along . After they get back , Jane and Rafael share a kiss . 72 8 `` Chapter Seventy - Two '' Melanie Mayron Carolina Rivera & Paul Sciarrotta January 26 , 2018 ( 2018 - 01 - 26 ) 0.68 Mateo starts kindergarten and invites a friend over to Xiomara and Rogelio 's house . Jane and Rafael try to cover up that they are pretending to live in their house so they do n't get kicked out . Jane and Rafael show mixed emotions following the kiss and try again , which Mateo notices . He then proclaims : `` Does this mean you 're going to get married ? '' After Xo convinces Rogelio to attend couples therapy , they both are surprised by their emotions . Meanwhile , Petra hires a lawyer named Jane Ramos ( Rosario Dawson ) , to ensure she gets no charges pending Anezka 's death . Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 , 2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date . Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow . Rogelio is convinced he has male post-partum depression , which puts him at odds with with celebrity Post-Partum Advocate River Fields ( guest star Brooke Shields ) on The Talk . Also , Alba finally confides in Jane why she turned down the marriage proposal , leaving Jane to help Alba in the romance department when Jane takes Alba to a sex store and buys her a vibrator . Elsewhere , Jane and Rafael agree to have sex , and Petra questions her own sexual identity when she starts to have sex dreams about Jane `` J.R. '' Ramos after the lawyer begins flirting with her . 75 11 `` Chapter Seventy - Five '' Gina Lamar Merigan Mulhern March 2 , 2018 ( 2018 - 03 - 02 ) 0.59 Jane is ready to move on from the Marbella Hotel issues and pursue writing full time , but an unexpected development forces her to consider returning to her teaching roots . Xiomara decides to go back to school where she listens in on an art history class , while Jane teaches a seminar about writing . Here , Jane finds out Jonathan Chavez , her former advisor , is taking advantage of another student of his . While trying to warn the student , Jane accidentally grabs her breasts to avoid falling over and , as a result , does n't get the teaching job she was applying for . Meanwhile , Petra gets arrested for Anezka 's murder due to the blackmailer planting the screws in her apartment , leading Jane Ramos to tell Petra that she was framing her because she was being threatened by an unknown blackmailer . Also , Rafael and Jane continue their affair . Elsewhere , Rogelio tries to get Eva Longoria to join his Passions of Santos reboot , but instead finds out the two are distant cousins . 76 12 `` Chapter Seventy - Six '' Melanie Mayron Leah Longoria March 9 , 2018 ( 2018 - 03 - 09 ) 0.58 Jane 's fixation on a negative review has left her with writer 's block , so Rogelio suggests she take a writing class to help . With the encouragement of Jane , Rafael is ready to dig into his past to find out who his real biological parents are . Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 ) 0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind . Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R 's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65 Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider . Alba is frustrated with Rogelio when it seems he is once again making his career a priority over Xo 's well being . Meanwhile , Petra turns to Rafael for advice on her love life after he finally learns about her new romance with J.R. 79 15 `` Chapter Seventy - Nine '' Gina Lamar Paul Sciarrotta April 6 , 2018 ( 2018 - 04 - 06 ) 0.53 80 16 `` Chapter Eighty '' Micah Schraft Micah Schraft April 13 , 2018 ( 2018 - 04 - 13 ) 0.57 Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out . 81 17 `` Chapter Eighty - One '' Gina Lamar Jennie Snyder Urman & Paul Sciarrotta April 20 , 2018 ( 2018 - 04 - 20 ) 0.58 Rafael intends to propose to Jane , however Rose 's news unsettles him . Alba becomes a citizen and the Villanueva 's throw her a surprise party . Jane Ramos and Petra plan to move in together , until Jane finds out that Petra killed Anezka . As she walks out , Krishna tells her that her life is in danger . J.R. saves Petra 's life by shooting , and killing , the blackmailer . As Jane meets Rafael , Michael is revealed to be alive .
  Fact 1.0: The listing has columns for overall episode number, season episode number, title, director, writers, original air date, and U.S. viewers in millions.
  Fact 1.1: Chapter Sixty-Five is episode 65 overall, directed by Brad Silberling and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.2: Chapter Sixty-Five first aired October 13, 2017, with 0.68 million U.S. viewers.
  Fact 1.3: Chapter Sixty-Five picks up immediately after where Season 3 left off.
  Fact 1.4: Jane is reunited with her first love, Adam, who found the love letter that Michael wrote.
  Fact 1.5: Jane is surprised but happy to see Adam.
  Fact 1.6: Xo and Alba are less thrilled that Adam is back.
  Fact 1.7: Rafael loses ownership of the hotel to his sister.
  Fact 1.8: As a result, Rafael finds himself living at Jane's house, broke and homeless.
  Fact 1.9: Jane and Rafael learn that the missing Petra has been found after she got away from Anezka.
  Fact 1.10: Jane and Rafael rush to Petra's side, but Anezka is nowhere to be found.
  Fact 1.11: When Petra returns to the hotel, Luisa thinks Petra is Anezka.
  Fact 1.12: Petra goes along with the game until the real Anezka returns.
  Fact 1.13: Xo and Rogelio's married life is interrupted when they confront the realization that Darci is having Rogelio's baby.
  Fact 1.14: Darci wants full custody of the baby.
  Fact 1.15: Chapter Sixty-Six is directed by Gina Lamar and written by Valentina Garza, Jessica O'Toole and Amy Rardin.
  Fact 1.16: Chapter Sixty-Six first aired October 20, 2017, with 0.61 million U.S. viewers.
  Fact 1.17: Jane's relationship with Adam helps bring out her 'fun side'.
  Fact 1.18: Adam's relationship with Jane reminds her that she is not just a mom.
  Fact 1.19: Adam gets a job opportunity out of state.
  Fact 1.20: Adam asks Jane's opinion on whether he should stay or go.
  Fact 1.21: Rogelio continues to face issues with his co-star at work.
  Fact 1.22: Rogelio's issues with his co-star are about to get even worse.
  Fact 1.23: Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby.
  Fact 1.24: Petra and Rafael begin working together to try to buy back the hotel from Luisa.
  Fact 1.25: Petra and Rafael run into trouble involving Anezka.
  Fact 1.26: Petra and Rafael's trouble involving Anezka causes tension between them.
  Fact 1.27: Chapter Sixty-Seven is directed by Fernando Sariñana and written by Carolina Rivera and Micah Schraft.
  Fact 1.28: Chapter Sixty-Seven first aired October 27, 2017, with 0.60 million U.S. viewers.
  Fact 1.29: Jane and Adam's relationship seems to be moving on.
  Fact 1.30: Jane thinks it might be time for Adam to meet her son Mateo.
  Fact 1.31: Rafael is not happy with the idea of Adam meeting Mateo.
  Fact 1.32: As the arrival of Darci's baby approaches, Rogelio's jealousy of Darci's new boyfriend increases.
  Fact 1.33: Xo encourages Rogelio to step up and be the father he was meant to be.
  Fact 1.34: Rafael is determined to get his hotel and life back from Luisa.
  Fact 1.35: Rafael will do whatever he has to do to get his hotel and life back, with or without Petra's help.
  Fact 1.36: Chapter Sixty-Eight is directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.37: Chapter Sixty-Eight first aired November 3, 2017, with 0.69 million U.S. viewers.
  Fact 1.38: Jane becomes angry when she learns Rafael introduced Mateo to his new girlfriend, Katherine, without her consent.
  Fact 1.39: Jane thinks Katherine is a negative influence on both Rafael and Mateo.
  Fact 1.40: Jane turns to Petra for help with the situation.
  Fact 1.41: Rogelio and Darci struggle to choose a name for the baby.
  Fact 1.42: Xo is focused on opening her own dance studio.
  Fact 1.43: Xo decides to go into business with an unlikely foe.
  Fact 1.44: Chapter Sixty-Nine is episode 69 overall and season 5, directed by Stuart Gillard and written by Paul Sciarrotta.
  Fact 1.45: Chapter Sixty-Nine first aired November 10, 2017, with 0.65 million U.S. viewers.
  Fact 1.46: After Adam reveals his dating past to Jane, Jane becomes unsure and insecure.
  Fact 1.47: Jane's insecurity puts a strain on her relationship with Adam.
  Fact 1.48: Lina seeks Jane's advice about whether she should get married.
  Fact 1.49: Petra hears important information that could help Rafael get the hotel back.
  Fact 1.50: Xo asks Rogelio to consider having a vasectomy.
  Fact 1.51: Rogelio is very hesitant about the vasectomy.
  Fact 1.52: Chapter Seventy is episode 70 overall and season 6, directed by Melanie Mayron and written by Carolina Rivera and Micah Schraft.
  Fact 1.53: Chapter Seventy first aired November 17, 2017, with 0.61 million U.S. viewers.
  Fact 1.54: Jane's book is finally being released.
  Fact 1.55: Jane's plans to have a party at the local bookshop where she was inspired to be a writer hit a snag.
  Fact 1.56: Rogelio devises a plan to save the day for Jane.
  Fact 1.57: One of Rogelio's arch-rivals threatens to ruin the day.
  Fact 1.58: Rafael gets a new job, but it is not exactly what he was hoping for.
  Fact 1.59: Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa.
  Fact 1.60: Rafael wants nothing to do with Petra's scheme.
  Fact 1.61: Chapter Seventy-One is episode 71 overall and season 7, directed by Micah Schraft and written by Valentina Garza and Deidre Shaw.
  Fact 1.62: Chapter Seventy-One first aired December 8, 2017, with 0.65 million U.S. viewers.
  Fact 1.63: Anezka is revealed to have faked her death to see where her loyalties lie.
  Fact 1.64: Anezka initially trusts Petra after believing she saw Petra cry.
  Fact 1.65: Anezka threatens Petra after finding out Petra was not crying because of Anezka's fake death.
  Fact 1.66: Later, Anezka falls from Petra's apartment to her death, for real.
  Fact 1.67: Xiomara and Jane set off on Jane's book tour.
  Fact 1.68: Alba decides to tag along on Jane's book tour.
  Fact 1.69: After they get back, Jane and Rafael share a kiss.
  Fact 1.70: Chapter Seventy-Two is episode 72 overall and season 8, directed by Melanie Mayron and written by Carolina Rivera and Paul Sciarrotta.
  Fact 1.71: Chapter Seventy-Two first aired January 26, 2018, with 0.68 million U.S. viewers.
  Fact 1.72: Mateo starts kindergarten.
  Fact 1.73: Mateo invites a friend over to Xiomara and Rogelio's house.
  Fact 1.74: Jane and Rafael try to cover up that they are pretending to live in their house so they don't get kicked out.
  Fact 1.75: Jane and Rafael show mixed emotions following the kiss and try again, which Mateo notices.
  Fact 1.76: Mateo proclaims: 'Does this mean you're going to get married?'
  Fact 1.77: After Xo convinces Rogelio to attend couples therapy, Xo and Rogelio are both surprised by their emotions.
  Fact 1.78: Petra hires a lawyer named Jane Ramos, played by Rosario Dawson.
  Fact 1.79: Petra hires Jane Ramos to ensure Petra gets no charges pending Anezka's death.
  Fact 1.80: Rafael moves out of Jane's house into his new apartment.
  Fact 1.81: Chapter Seventy-Three is episode 73 overall and season 9, directed by Eric Lea and written by Valentina Garza and Chantelle M. Wells.
  Fact 1.82: Chapter Seventy-Three first aired February 2, 2018, with 0.67 million U.S. viewers.
  Fact 1.83: Jane and Rafael realize that the uncertain state of their relationship is stressing out Mateo.
  Fact 1.84: Jane and Rafael decide that they need to make a decision.
  Fact 1.85: Petra is on the outs with Jane and Rafael after the Luisa fiasco.
  Fact 1.86: Petra tries to figure out a way to get back into Jane and Rafael's good graces.
  Fact 1.87: Rogelio and Xo are both contemplating their career choices.
  Fact 1.88: Chapter Seventy-Four is episode 74 overall and season 10, directed by Gina Rodriguez and written by Micah Schraft and Paul Sciarrotta.
  Fact 1.89: Chapter Seventy-Four first aired February 9, 2018, with 0.80 million U.S. viewers.
  Fact 1.90: Jane and Rafael try to be friends, which works great until Jane goes on a date.
  Fact 1.91: Petra and Jane's new partnership is off to a rocky start.
  Fact 1.92: Petra and Jane both know they have to find a way to work together.
  Fact 1.93: Rogelio is convinced he has male post-partum depression.
  Fact 1.94: Rogelio's belief puts him at odds with celebrity Post-Partum Advocate River Fields on The Talk.
  Fact 1.95: River Fields is guest starred by Brooke Shields.
  Fact 1.96: Alba finally confides in Jane why she turned down the marriage proposal.
  Fact 1.97: Jane helps Alba in the romance department.
  Fact 1.98: Jane takes Alba to a sex store and buys her a vibrator.
  Fact 1.99: Jane and Rafael agree to have sex.
  Fact 1.100: Petra questions her own sexual identity.
  Fact 1.101: Petra starts to have sex dreams about Jane 'J.R.' Ramos after the lawyer begins flirting with her.
  Fact 1.102: Chapter Seventy-Five is episode 75 overall and season 11, directed by Gina Lamar and written by Merigan Mulhern.
  Fact 1.103: Chapter Seventy-Five first aired March 2, 2018, with 0.59 million U.S. viewers.
  Fact 1.104: Jane is ready to move on from the Marbella Hotel issues and pursue writing full time.
  Fact 1.105: An unexpected development forces Jane to consider returning to her teaching roots.
  Fact 1.106: Xiomara decides to go back to school, where she listens in on an art history class.
  Fact 1.107: Jane teaches a seminar about writing.
  Fact 1.108: Jane finds out Jonathan Chavez, her former advisor, is taking advantage of another student.
  Fact 1.109: While trying to warn the student, Jane accidentally grabs her breasts to avoid falling over.
  Fact 1.110: Because of this, Jane does not get the teaching job she was applying for.
  Fact 1.111: Petra gets arrested for Anezka's murder due to the blackmailer planting the screws in Petra's apartment.
  Fact 1.112: Jane Ramos tells Petra that she was framing Petra because Jane was being threatened by an unknown blackmailer.
  Fact 1.113: Rafael and Jane continue their affair.
  Fact 1.114: Rogelio tries to get Eva Longoria to join his Passions of Santos reboot.
  Fact 1.115: Rogelio finds out that he and Eva Longoria are distant cousins.
  Fact 1.116: Chapter Seventy-Six is episode 76 overall and season 12, directed by Melanie Mayron and written by Leah Longoria.
  Fact 1.117: Chapter Seventy-Six first aired March 9, 2018, with 0.58 million U.S. viewers.
  Fact 1.118: Jane's fixation on a negative review has left her with writer's block.
  Fact 1.119: Rogelio suggests Jane take a writing class to help.
  Fact 1.120: With Jane's encouragement, Rafael is ready to dig into his past to find out who his real biological parents are.
  Fact 1.121: Rogelio is trying to make things less about him and more about Xo.
  Fact 1.122: Rogelio grows suspicious when he learns Xo is keeping a secret from him.
  Fact 1.123: The secret Xo has been keeping is not what Rogelio suspected.
  Fact 1.124: Petra tests the loyalty of her lawyer, Jane Ramos, by pretending to be romantically interested in her.
  Fact 1.125: Petra is trying to find the identity of the blackmailer.
  Fact 1.126: Petra confides in Jane R. about her feelings, which are not fake.
  Fact 1.127: Petra and Jane R. sleep together for the first time.
  Fact 1.128: Chapter Seventy-Seven is episode 77 overall and season 13, directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.129: Chapter Seventy-Seven first aired March 16, 2018, with 0.59 million U.S. viewers.
  Fact 1.130: Jane, Rogelio and Alba learn that Xo might have breast cancer.
  Fact 1.131: Jane, Rogelio and Alba try to comfort Xo.
  Fact 1.132: Xo insists they let her live her life.
  Fact 1.133: Rogelio finally gets River Fields to agree to meet with him.
  Fact 1.134: Rogelio learns that River has another plan in mind.
  Fact 1.135: Rafael is on a mission to find his sister, who may have information he has been looking for about his origins.
  Fact 1.136: Rafael visits Rose in prison to ask for her help.
  Fact 1.137: Rafael quickly turns Rose down when she demands to know Luisa's location.
  Fact 1.138: Jane is upset with Petra after Jane's twin girls spill a secret to Mateo about the Tooth Fairy being make-believe.
  Fact 1.139: Jane R. Ramos tells Petra she is not interested in a relationship after their romantic night together.
  Fact 1.140: The blackmailer paid a visit to Jane R.'s dementia-stricken mother in a rest home.
  Fact 1.141: The identity of Jane R.'s blackmailer is finally revealed at the end of the episode.
  Fact 1.142: Chapter Seventy-Eight is episode 78 overall and season 14, directed by Justin Baldoni and written by Valentina Garza and Micah Schraft.
  Fact 1.143: Chapter Seventy-Eight first aired March 23, 2018, with 0.65 million U.S. viewers.
  Fact 1.144: Xo struggles to make an important decision on whether or not to get a full mastectomy.
  Fact 1.145: Xo looks to Jane for guidance on the mastectomy decision.
  Fact 1.146: Xo's decision leaves Rogelio feeling like an outsider.
  Fact 1.147: Alba is frustrated with Rogelio when he once again seems to make his career a priority over Xo's well-being.
  Fact 1.148: Petra turns to Rafael for advice on her love life.
  Fact 1.149: Rafael finally learns about Petra's new romance with J.R.
  Fact 1.150: Chapter Seventy-Nine is episode 79 overall and season 15, directed by Gina Lamar and written by Paul Sciarrotta.
  Fact 1.151: Chapter Seventy-Nine first aired April 6, 2018, with 0.53 million U.S. viewers.
  Fact 1.152: Chapter Eighty is episode 80 overall and season 16, directed by Micah Schraft and written by Micah Schraft.
  Fact 1.153: Chapter Eighty first aired April 13, 2018, with 0.57 million U.S. viewers.
  Fact 1.154: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 1.155: Jane and Rafael's plan falls apart when Mateo breaks an antique $4,000 clock.
  Fact 1.156: Rafael moves into the Villanueva house.
  Fact 1.157: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 1.158: Rafael moves back out after the argument.
  Fact 1.159: Xiomara starts chemotherapy.
  Fact 1.160: Xiomara makes a friend named Donna.
  Fact 1.161: The timeline moves forward nine weeks.
  Fact 1.162: Donna dies of unforeseen consequences.
  Fact 1.163: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 1.164: Alba and Rafael are still in a fight.
  Fact 1.165: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 1.166: Alba breaks down to Jane.
  Fact 1.167: Alba and Rafael make up.
  Fact 1.168: Xiomara bonds with River over telenovelas.
  Fact 1.169: Xiomara accidentally eats a pot brownie.
  Fact 1.170: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 1.171: Jane Ramos gets herself disbarred.
  Fact 1.172: As a result, Jane Ramos gets the case against Petra thrown out.
  Fact 1.173: Chapter Eighty-One is episode 81 overall and season 17, directed by Gina Lamar and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.174: Chapter Eighty-One first aired April 20, 2018, with 0.58 million U.S. viewers.
  Fact 1.175: Rafael intends to propose to Jane.
  Fact 1.176: Rose's news unsettles Rafael.
  Fact 1.177: Alba becomes a citizen.
  Fact 1.178: The Villanueva family throws Alba a surprise party.
  Fact 1.179: Jane Ramos and Petra plan to move in together.
  Fact 1.180: Jane finds out Petra killed Anezka.
  Fact 1.181: As Jane walks out, Krishna tells her that her life is in danger.
  Fact 1.182: J.R. saves Petra's life by shooting and killing the blackmailer.
  Fact 1.183: As Jane meets Rafael, Michael is revealed to be alive.

Retrieved items:
<item 1>
Jane the Virgin (season 4)
The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and
Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make
up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin
American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out . 81 17 `` Chapter Eighty - One '' Gina Lamar Jennie
Snyder Urman & Paul Sciarrotta April 20 , 2018 ( 2018 - 04 - 20 ) 0.58 Rafael intends to propose to Jane , however Rose 's news unsettles him . Alba becomes a
citizen and the Villanueva 's throw her a surprise party . Jane Ramos and Petra plan to move in together , until Jane finds out that Petra killed Anezka .
</item 1>

<item 2>
Jane the Virgin (season 4)
Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for
her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret
to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not
interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R
's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65
Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider
.
</item 2>

<item 3>
Jane the Virgin (season 4)
Here , Jane finds out Jonathan Chavez , her former advisor , is taking advantage of another student of his . While trying to warn the student , Jane
accidentally grabs her breasts to avoid falling over and , as a result , does n't get the teaching job she was applying for . Meanwhile , Petra gets arrested
for Anezka 's murder due to the blackmailer planting the screws in her apartment , leading Jane Ramos to tell Petra that she was framing her because she was
being threatened by an unknown blackmailer . Also , Rafael and Jane continue their affair . Elsewhere , Rogelio tries to get Eva Longoria to join his Passions
of Santos reboot , but instead finds out the two are distant cousins . 76 12 `` Chapter Seventy - Six '' Melanie Mayron Leah Longoria March 9 , 2018 ( 2018 - 03
- 09 ) 0.58 Jane 's fixation on a negative review has left her with writer 's block , so Rogelio suggests she take a writing class to help . With the
encouragement of Jane , Rafael is ready to dig into his past to find out who his real biological parents are .
</item 3>

<item 4>
Jane the Virgin (season 4)
Rogelio is convinced he has male post-partum depression , which puts him at odds with with celebrity Post-Partum Advocate River Fields ( guest star Brooke
Shields ) on The Talk . Also , Alba finally confides in Jane why she turned down the marriage proposal , leaving Jane to help Alba in the romance department
when Jane takes Alba to a sex store and buys her a vibrator . Elsewhere , Jane and Rafael agree to have sex , and Petra questions her own sexual identity when
she starts to have sex dreams about Jane `` J.R. '' Ramos after the lawyer begins flirting with her . 75 11 `` Chapter Seventy - Five '' Gina Lamar Merigan
Mulhern March 2 , 2018 ( 2018 - 03 - 02 ) 0.59 Jane is ready to move on from the Marbella Hotel issues and pursue writing full time , but an unexpected
development forces her to consider returning to her teaching roots . Xiomara decides to go back to school where she listens in on an art history class , while
Jane teaches a seminar about writing .
</item 4>

<item 5>
Jane the Virgin (season 4)
However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself
living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her
side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real
Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants
full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's
relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom .
</item 5>

<item 6>
Jane the Virgin (season 4)
Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 ,
2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they
decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to
get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez
Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date .
Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow .
</item 6>

<item 7>
Jane the Virgin (season 4)
Ramos Alfonso DiLuca as Jorge Garcia Special Guest Star ( edit ) Brooke Shields as River Fields Guest ( edit ) Francisco San Martin as Fabian Regalo del Cielo
Diane Guerrero as Lina Santillan Sharon Osbourne as herself Julie Chen as herself Sara Gilbert as herself Sheryl Underwood as herself Eve as herself Melanie
Mayron as Marlene Donaldson Eva Longoria as herself Adam Rodriguez as Jonathan Chavez Amy Brenneman as Donna Episodes ( edit ) See also : List of Jane the
Virgin episodes No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 65 `` Chapter Sixty - Five '' Brad
Silberling Jennie Snyder Urman & Paul Sciarrotta October 13 , 2017 ( 2017 - 10 - 13 ) 0.68 Picking up immediately after where Season 3 left off , Jane is
reunited with her first love , Adam , who found the love letter that Michael wrote . She is surprised , but happy , to see him . However , both Xo and Alba are
less thrilled that he is back .
</item 7>

<item 8>
Jane the Virgin (season 4)
69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she
becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not .
Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy )
done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's
book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help
Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day .
</item 8>

<item 9>
Jane the Virgin (season 4)
Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But
the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically
interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of
them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 )
0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live
her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind .
</item 9>

<item 10>
Jane the Virgin (season 4)
Contents ( hide ) 1 Cast and characters 1.1 Main 1.2 Recurring 1.3 Special Guest Star 1.4 Guest 2 Episodes 3 Ratings 4 References Cast and characters ( edit )
Main ( edit ) Gina Rodriguez as Jane Gloriana Villanueva Andrea Navedo as Xiomara `` Xo '' Gloriana de la Vega Yael Grobglas as Petra Solano / Anezka Archuletta
Justin Baldoni as Rafael Solano Ivonne Coll as Alba Gloriana Garcia Elias Janssen as Mateo Gloriano Rogelio Solano Villanueva Jaime Camil as Rogelio de la Vega
Anthony Mendez as the Latin Lover Narrator ( voice over ) Recurring ( edit ) Tyler Posey as Adam Alvaro Yara Martinez as Dr. Luisa Alver Justina Machado as
Darci Factor Bridget Regan as Rose Solano Priscilla Barnes as Magda Shelly Bhalla as Krishna Rosario Dawson as Jane `` J.R. ''
</item 10>
=== END ITEM id=a193bad5cd116ae7 ===

=== ITEM id=0ede4436d1037f04 ===
Gold sentences and facts:
Sentence 0: The Battle of Lake Erie , sometimes called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval battles of the War of 1812 .
  Fact 0.0: The Battle of Lake Erie is sometimes called the Battle of Put-in-Bay.
  Fact 0.1: The Battle of Lake Erie was fought on 10 September 1813.
  Fact 0.2: The Battle of Lake Erie was fought in Lake Erie off the coast of Ohio.
  Fact 0.3: The Battle of Lake Erie was fought during the War of 1812.
  Fact 0.4: Nine vessels of the United States Navy defeated six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.5: Nine vessels of the United States Navy captured six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.6: The Battle of Lake Erie ensured American control of the lake for the rest of the War of 1812.
  Fact 0.7: The Battle of Lake Erie allowed the Americans to recover Detroit.
  Fact 0.8: The Battle of Lake Erie allowed the Americans to win the Battle of the Thames.
  Fact 0.9: Winning the Battle of the Thames broke the Indian confederation of Tecumseh.
  Fact 0.10: The Battle of Lake Erie was one of the biggest naval battles of the War of 1812.

Retrieved items:
<item 1>
Battle of Lake Erie
- 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Battle_of_Lake_Erie&oldid=799568331 '' Categories : Conflicts in 1813 Battles involving
Canada History of Ohio Naval battles of the War of 1812 Naval battles involving Canada Lake Erie 1813 in Ohio September 1813 events Hidden categories : CS1
maint : Extra text : authors list Pages using deprecated image syntax Articles needing additional references from September 2017 All articles needing additional
references Wikipedia articles incorporating a citation from The American Cyclopaedia Wikipedia articles incorporating a citation from The American Cyclopaedia
with a Wikisource reference Coordinates not on Wikidata Use dmy dates from May 2011 Talk Contents About Wikipedia Afrikaans Català Deutsch Español Français 한국어
Magyar Bahasa Melayu 日本 語 Polski Português ไทย 中文 Edit links This page was last edited on 8 September 2017 , at 14 : 22 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply .
</item 1>

<item 2>
Battle of Lake Erie
Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St. Philip 2nd Fort Bowyer Pacific Ocean James Island Charles Island Nuku Hiva Downes
Expedition Porter Expedition Typee Valley Valparaiso ( Capture of USS Essex ) Seringapatam Mutiny Action of 9 May 1814 The Battle of Lake Erie , sometimes
called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United
States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn
allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval
battles of the War of 1812 .
</item 2>

<item 3>
Battle of Lake Erie
( 2001 ) . The Naval Institute historical atlas of the U.S. Navy . Naval Institute Press . p. 48 . ISBN 978 - 1 - 55750 - 984 - 0 . Jump up ^ Hickey , Donald R.
Do n't Give Up The Ship , p. 355 . Jump up ^ Roosevelt 2004 , p. 152 . Jump up ^ William James The Naval History of Great Britain , from the Declaration of War
by Trace in 1793 , to the Accession of George IV . Volume 6 Richard Bentley 1847 Page 252 References ( edit ) Altoff , Gerard T. ( 1999 ) . Oliver Hazard Perry
and the Battle of Lake Erie . Put - in - Bay , OH : The Perry Group . ISBN 978 - 1887794039 . Copes , Jan M. ( Fall 1994 ) . `` The Perry Family : A Newport
Naval Dynasty of the Early Republic '' . Newport History : Bulletin of the Newport Historical Society . Newport , RI : Newport Historical Society .
</item 3>

<item 4>
Battle of Lake Erie
External links ( edit ) Wikimedia Commons has media related to Battle of Lake Erie . United States Navy portal Rhode Island portal Erie portal Ohio portal War
of 1812 portal History of Canada portal Wikiquote has quotations related to : Battle of Lake Erie `` Log of the Battle of Lake Erie '' by Sailing Master William
Taylor US Brig Niagara Barclay 's court martial records and correspondence Battle of Lake Erie Historical Sources Animated history of the Battle of Lake Erie ``
Erie , Lake , Battle of '' . The American Cyclopædia. 1879 . Conflicts of the War of 1812 Battles of the War of 1812 United States Washington , D.C. Burning of
Washington Georgia Battle of Fort Peter Louisiana Battle of New Orleans Siege of Fort St. Philip Maryland Battle of Baltimore Battle of Bladensburg Battle of
Caulk 's Field Battle of North Point Battle of St.
</item 4>

<item 5>
Battle of Lake Erie
Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS Guerriere Capture of HMS Frolic USS United States vs HMS
Macedonian USS Constitution vs HMS Java Sinking of HMS Peacock Rappahannock River Capture of USS Chesapeake Capture of HMS Dominica Capture of USS Argus Capture
of HMS Boxer Capture of USS Frolic Capture of HMS Epervier Sinking of HMS Reindeer Sinking of HMS Avon Fayal Capture of USS President Capture of HMS Cyane and
HMS Levant Capture of HMS Penguin Capture of East India Company ship Nautilus East Coast Chesapeake Bay Alexandria Baltimore Hampden Fort Peter Great Lakes /
Saint Lawrence River Lake Ontario 1st Sacket 's Harbor York Fort George 2nd Sacket 's Harbor Lake Erie Fort Oswego Lake Huron Lake Champlain West Indies / Gulf
Coast La Guaira 1st Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St.
</item 5>

<item 6>
Battle of Lake Erie
Battle of Lake Erie - wikipedia Battle of Lake Erie Jump to : navigation , search Battle of Lake Erie Part of the War of 1812 Battle of Lake Erie by William
Henry Powell , painted 1865 , shows Oliver Hazard Perry transferring from Lawrence to Niagara Date 10 September 1813 Location Lake Erie , near Put - in - Bay ,
Ohio Result Decisive American victory Belligerents United States United Kingdom Commanders and leaders Oliver Hazard Perry Jesse Elliot Robert Heriot Barclay
Robert Finnis Strength 5 schooners 3 brigs 1 sloop 2 ships 1 brig 2 schooners 1 sloop Casualties and losses 27 killed 96 wounded 1 brig severely damaged 41
killed 93 wounded 306 captured entire squadron captured Detroit frontier Tippecanoe Fort Mackinac ( 1812 ) Brownstown Maguaga Fort Dearborn Detroit Fort
Harrison Fort Wayne Wild Cat Creek Mississinewa Frenchtown Africa Point Fort Meigs Fort Stephenson Lake Erie Thames Longwoods Prairie du Chien Campbell Island
Mackinac Island ( 1814 ) Lake Huron Malcolm 's Mills Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS
</item 6>

<item 7>
Battle of Lake Erie
ISBN 0 - 8117 - 2876 - 5 . Roosevelt , Theodore ( 2004 ) . The Naval War of 1812 . New York : The Modern Library . ISBN 0 - 375 - 75419 - 9 . Skaggs , David ;
Atloff , Gerard ( 1997 ) . A Signal Victory : The Lake Erie Campaign , 1812 -- 13 . Naval Institute Press . ISBN 1 - 55750 - 892 - 5 . Symonds , Craig ( 2005 )
. Decision at Sea . Oxford University Press . ISBN 0 - 19 - 517145 - 4 . Zaslow , Morris ( ed ) . ( 1964 ) . The Defended Border : Upper Canada and the War of
1812 . Toronto : Macmillan of Canada . ISBN 0 - 7705 - 1242 - 9 . Further reading ( edit ) Axelrod , Alen ; Phillips , Charles . The MacMillan Dictionary of
Military Biography ( New York : MacMillan , 1998 . )
</item 7>

<item 8>
Battle of Lake Erie
It was one of the biggest naval battles of the War of 1812 . Contents ( hide ) 1 Background 1.1 1812 1.2 1813 1.3 Blockades of Presque Isle and Amherstburg 2
Battle 3 Casualties 4 Aftermath 5 Reconstructions and memorials 6 Reasons for the American victory 7 Order of battle 8 Footnotes 9 Notes 10 References 11
Further reading 12 External links Background ( edit ) 1812 ( edit ) When the war broke out , the British immediately seized control of Lake Erie . They already
had a small force of warships there : the sloop - of - war Queen Charlotte and the brig General Hunter . The schooner Lady Prevost was under construction and
was put into service a few weeks after the outbreak of war . These vessels were controlled by the Provincial Marine , which was a military transport service and
not a naval service . Nevertheless , the Americans lacked any counter to the British armed vessels .
</item 8>

<item 9>
Battle of Lake Erie
USS United States vs HMS Macedonian Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie Battle of Fort Oswego Engagements on Lake Huron
Engagements on Lake Ontario First Battle of Sacket 's Harbor Gulf Coast Action of 13 December 1814 Battle of Lake Borgne Pacific Ocean Action off James Island
Action off Charles Island Nuku Hiva Campaign Battle of Valparaiso ( Capture of USS Essex ) See also : American Indian Wars , Creek War , Napoleonic Wars , and
Tecumseh 's War Portal definition textbooks quotes source texts media news stories Coordinates : 41 ° 39 ′ 44 '' N 82 ° 49 ′ 30 '' W ﻿ / ﻿ 41.66222 ° N 82.82500
° W ﻿ / 41.66222 ; - 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Battle of Lake Erie
James Massacre Missouri Battle of Credit Island Battle of the Sink Hole British North America Lower Canada Battle of the Chateauguay First Battle of Lacolle
Mills Second Battle of Lacolle Mills Upper Canada Battle of Beaver Dams Battle of Chippawa Battle of Cook 's Mills Battle of Crysler 's Farm Battle of Fort
George Battle of Frenchman 's Creek Battle of Longwoods Battle of Lundy 's Lane Battle of Malcolm 's Mills Battle of Queenston Heights Battle of Stoney Creek
Battle of the Thames Battle of York Capture of Fort Erie Raid on Elizabethtown Raid on Port Dover Raid on Gananoque Siege of Fort Erie Spanish Empire Spanish
Florida Battle of Pensacola Naval battles Atlantic Ocean Capture of HMS Boxer Capture of HMS Cyane Capture of HMS Epervier Capture of HMS Frolic Capture of HMS
Penguin Capture of HMS Dominica Capture of USS Argus Capture of USS Chesapeake Capture of USS President Chesapeake Bay Flotilla USS Constitution vs HMS Java
Sinking of HMS Avon Battle of Fayal Sinking of HMS Peacock Sinking of HMS Reindeer USS Constitution vs HMS Guerriere USS United States vs HMS Macedonian
Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie
</item 10>
=== END ITEM id=0ede4436d1037f04 ===

=== ITEM id=7ae9820ca04712d5 ===
Gold sentences and facts:
Sentence 0: The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle Eastern politics . Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League from 1979 until 1989 .
  Fact 0.0: Time has elapsed since the Camp David Accords.
  Fact 0.1: The passage of time has left no doubt about the enormous ramifications of the Camp David Accords on Middle Eastern politics.
  Fact 0.2: The perception of Egypt within the Arab world changed.
  Fact 0.3: The change in perception of Egypt was the most notable effect.
  Fact 0.4: Egypt had the most powerful of the Arab militaries.
  Fact 0.5: Egypt had a history of leadership in the Arab world under Nasser.
  Fact 0.6: Egypt had more leverage than any other Arab state to advance Arab interests.
  Fact 0.7: Egypt was subsequently suspended from the Arab League.
  Fact 0.8: Egypt's suspension from the Arab League lasted from 1979 until 1989.

Retrieved items:
<item 1>
Camp David Accords
Camp David Accords - wikipedia Camp David Accords Jump to : navigation , search Camp David Accords Framework for Peace in the Middle East and Framework for the
Conclusion of a Peace Treaty between Egypt and Israel Celebrating the signing of the Camp David Accords : Menachem Begin , Jimmy Carter , Anwar Sadat Type
Bilateral treaty Signed 17 September 1978 ( 1978 - 09 - 17 ) Location Washington , DC , United States Original signatories Egypt Israel Signatories Menachem
Begin ( Prime Minister of Israel ) Anwar Sadat ( President of Egypt ) Jimmy Carter ( President of the United States ) Ratifiers Egypt Israel Language English
French The Camp David Accords were signed by Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin on 17 September 1978 , following twelve
days of secret negotiations at Camp David . The two framework agreements were signed at the White House , and were witnessed by United States President Jimmy
Carter . The second of these frameworks ( A Framework for the Conclusion of a Peace Treaty between Egypt and Israel ) led directly to the 1979 Egypt -- Israel
Peace Treaty .
</item 1>

<item 2>
Camp David Accords
Accompanied by their capable negotiating teams and with their respective interests in mind , both leaders converged on Camp David for 13 days of tense and
dramatic negotiations from 5 to 17 September 1978 . By all accounts , Carter 's relentless drive to achieve peace and his reluctance to allow the two men to
leave without reaching an agreement are what played the decisive role in the success of the talks . Carter 's advisers insisted on the establishment of an
Egyptian - Israeli agreement which would lead to an eventual solution to the Palestine issue . They believed in a short , loose , and overt linkage between the
two countries amplified by the establishment of a coherent basis for a settlement . However , Carter felt they were not `` aiming high enough '' and was
interested in the establishment of a written `` land for peace '' agreement with Israel returning the Sinai Peninsula and West Bank . Numerous times both the
Egyptian and Israeli leaders wanted to scrap negotiations , only to be lured back into the process by personal appeals from Carter .
</item 2>

<item 3>
Camp David Accords
Meir Rosenne and Aharon Barak . Furthermore , the absence of the media contributed to the Accord 's successes : there were no possibilities provided to either
leader to reassure his political body or be driven to conclusions by members of his opposition . An eventual scrap of negotiations by either leader would have
proven disastrous , resulting in taking the blame for the summit 's failure as well as a disassociation from the White House . Ultimately , neither Begin nor
Sadat was willing to risk those eventualities . Both of them had invested enormous amounts of political capital and time to reach an agreement . Partial
agreements Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin acknowledge applause during a joint session of Congress in Washington , D.C.
, during which President Jimmy Carter announced the results of the Camp David Accords , 18 September 1978 . `` Remarks on the Signing of the Camp David Accords
'' Play media Jimmy Carter , seated with Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin , makes statements at a Joint session of the
United States Congress following the Camp David Accords . Problems playing this file ? See media help .
</item 3>

<item 4>
Camp David Accords
With the withdrawal , Israel also returned Egypt 's Abu - Rudeis oil fields in western Sinai , which contained long term , commercially productive wells . The
agreement also resulted in the United States committing to several billion dollars worth of annual subsidies to the governments of both Israel and Egypt ,
subsidies which continue to this day , and are given as a mixture of grants and aid packages committed to purchasing U.S. materiel . From 1979 ( the year of the
peace agreement ) to 1997 , Egypt received military aid of US $1.3 billion annually , which also helped modernize the Egyptian military . ( This is beyond
economic , humanitarian , and other aid , which has totaled more than US $25 billion . ) Eastern - supplied until 1979 , Egypt now received American weaponry
such as the M1A1 Abrams Tank , AH - 64 Apache gunship and the F - 16 fighter jet . In comparison , Israel has received $3 billion annually since 1985 in grants
and military aid packages . Consequences The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle
Eastern politics .
</item 4>

<item 5>
Camp David Accords
Hussein consequently felt diplomatically snubbed . One of Carter 's regrets was allowing Sadat to claim that he could speak for Hussein if Jordan refused to
join the talks , but by then the damage was done to the Jordanians . United States President Jimmy Carter greeting Egyptian President Anwar Sadat at the White
House shortly after the Camp David Accords went into effect , 8 April 1980 . The Camp David Accords also prompted the disintegration of a united Arab front in
opposition to Israel . Egypt 's realignment created a power vacuum that Saddam Hussein of Iraq , at one time only a secondary power , hoped to fill . Because of
the vague language concerning the implementation of Resolution 242 , the Palestinian problem became the primary issue in the Arab -- Israeli conflict
immediately following the Camp David Accords ( and , arguably , until today ) . Many of the Arab nations blamed Egypt for not putting enough pressure on Israel
to deal with the Palestinian problem in a way that would be satisfactory to them . Syria also informed Egypt that it would not reconcile with the nation unless
it abandoned the peace agreement with Israel .
</item 5>

<item 6>
Camp David Accords
Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab
world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League
from 1979 until 1989 . When the Camp David accords were signed , Jordan 's King Hussein saw it as a slap to the face when Sadat volunteered Jordan 's
participation in deciding how functional autonomy would work . More specifically , Sadat effectively said that Jordan would have a role in how the West Bank
would be administered . Like the Rabat Summit Resolution , the Camp David Accords circumscribed Jordan 's objective to reassert its control over the West Bank .
Focusing as it did on Egypt , the Carter administration accepted Sadat 's claim that he could deliver Hussein . However , with Arab world opposition building
against Sadat , Jordan could not risk accepting the Accords without the support from powerful Arab neighbours , like Iraq , Saudi Arabia , and Syria . Hussein
consequently felt diplomatically snubbed .
</item 6>

<item 7>
Camp David Accords
The issues of Israeli settlement withdrawal from the Sinai and the status of the West Bank created what seemed to be an impasse . In response , Carter had the
choice of trying to salvage the agreement by conceding the issue of the West Bank to Begin , while advocating Sadat 's less controversial position on the
removal of all settlements from the Sinai Peninsula . Or he could have refused to continue the talks , reported the reasons for their failure , and allowed
Begin to bear the brunt of the blame . Carter chose to continue and for three more days negotiated . During this time , Carter even took the two leaders to the
nearby Gettysburg National Military Park in the hopes of using the American Civil War as a simile to their own struggle . Consequently , the 13 days marking the
Camp David Accords were considered a success . Partly due to Carter 's determination in obtaining an Israeli - Egyptian agreement , a full two - week pledge to
a singular international problem . Additionally , Carter was beneficiary to a fully pledged American foreign team . Likewise , the Israeli delegation had a
stable of excellent talent in Ministers Dayan and Weizman and legal experts Dr.
</item 7>

<item 8>
Camp David Accords
According to The Continuum Political Encyclopedia of the Middle East : The normalization of relations ( between Israel and Egypt ) went into effect in January
1980 . Ambassadors were exchanged in February . The boycott laws were repealed by Egypt 's National Assembly the same month , and some trade began to develop ,
albeit less than Israel had hoped for . In March 1980 regular airline flights were inaugurated . Egypt also began supplying Israel with crude oil `` . According
to Kenneth Stein in Heroic Diplomacy : Sadat , Kissinger , Carter , Begin , and the Quest for Arab -- Israeli Peace : The Accords were another interim agreement
or step , but negotiations that flowed from the Accords slowed for several reasons . These included an inability to bring the Jordanians into the discussions ;
the controversy over settlements ; the inconclusive nature of the subsequent autonomy talks ; domestic opposition sustained by both Begin and Sadat and , in
Sadat 's case , ostracism and anger from the Arab world ; the emergence of a what became a cold peace between Egypt and Israel ; and changes in foreign policy
priorities including discontinuity in personnel committed to sustaining the negotiating process ( . )
</item 8>

<item 9>
Camp David Accords
The efforts initially focused on a comprehensive resolution of disputes between Israel and the Arab countries , gradually evolving into a search for a bilateral
agreement between Israel and Egypt . Upon assuming office on January 20 , 1977 , President Carter moved to rejuvenate the Middle East peace process that had
stalled throughout the 1976 presidential campaign in the United States . Following the advice of a Brookings Institution report , Carter opted to replace the
incremental , bilateral peace talks which had characterized Henry Kissinger 's shuttle diplomacy following the 1973 Yom Kippur War with a comprehensive ,
multilateral approach . The Yom Kippur War further complicated efforts to achieve the objectives written in United Nations Security Council Resolution 242 .
Israel 's Prime Minister Yitzhak Rabin and his successor , Menachem Begin , were both skeptical of an international conference . While Begin , who took office
in May 1977 , officially favored the reconvening of the conference , perhaps even more vocally than Rabin , and even accepted the Palestinian presence , in
actuality the Israelis and the Egyptians were secretly formulating a framework for bilateral talks .
</item 9>

<item 10>
Camp David Accords
Lastly , the biggest consequence of all may be in the psychology of the participants of the Arab - Israeli conflict . The success of Begin , Sadat , and Carter
at Camp David demonstrated to other Arab states and entities that negotiations with Israel were possible -- that progress results only from sustained efforts at
communication and cooperation . Despite the disappointing conclusion of the 1993 Oslo Accords between the PLO and Israel , and even though the 1994 Israel --
Jordan peace treaty has not fully normalized relations with Israel , both of these significant developments had little chance of occurring without the precedent
set by Camp David . Public support Although most Israelis supported the Accords , the Israeli settler movement opposed them because Sadat would not agree to a
treaty in which Israel had any presence in the Sinai Peninsula at all , Israel had to withdraw from the entire Sinai Peninsula . Israeli settlers tried to
prevent the government from dismantling their settlements . In Israel , there is lasting support of the Camp David Peace Accords , which have become a national
consensus , supported by 85 % of Israelis according to a 2001 poll taken by the Jaffee Center for Strategic Studies ( Israel based ) .
</item 10>
=== END ITEM id=7ae9820ca04712d5 ===
