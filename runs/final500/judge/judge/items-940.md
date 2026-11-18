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

This file has 7 items.

=== ITEM id=b2a934612f919643 ===
Gold sentences and facts:
Sentence 0: Group 3 in the periodic table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury ( element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson alkaline earth metals ← → group 4 IUPAC group number Name by element scandium group CAS group number ( US , pattern A-B-A ) IIIB old IUPAC number ( Europe , pattern A-B ) IIIA ↓ Period Scandium ( Sc ) 21 Transition metal 5 Yttrium ( Y ) 39 Transition metal 6 Lanthanum ( La * ) 57 Lanthanide 7 Actinium ( Ac * ) 89 Actinide * Whether the elements lutetium ( Lu ) and lawrencium ( Lr ) , in period 6 and 7 , are in group 3 is disputed . The grouping used in this article places La and Ac in group 3 , which is the most common form . For other groupings , see group 3 borders . Legend primordial element synthetic element Atomic number color : black = solid
  Fact 0.0: The periodic table has a group 3.
  Fact 0.1: The text lists the element names from Hydrogen through Oganesson.
  Fact 0.2: The text mentions alkaline earth metals.
  Fact 0.3: The article's grouping places lanthanum (La) and actinium (Ac) in group 3.
  Fact 0.4: The article says placing La and Ac in group 3 is the most common form.
  Fact 0.5: Scandium (Sc) has atomic number 21 and is a transition metal.
  Fact 0.6: Yttrium (Y) has atomic number 39, is in period 5, and is a transition metal.
  Fact 0.7: Lanthanum (La) has atomic number 57, is in period 6, and is a lanthanide.
  Fact 0.8: Actinium (Ac) has atomic number 89, is in period 7, and is an actinide.
  Fact 0.9: Whether lutetium (Lu) and lawrencium (Lr), in periods 6 and 7, are in group 3 is disputed.
  Fact 0.10: For other groupings, the article refers readers to group 3 borders.
  Fact 0.11: The scandium group has CAS group number IIIB (US pattern A-B-A).
  Fact 0.12: The scandium group has old IUPAC number IIIA (Europe pattern A-B).
  Fact 0.13: The article has a legend distinguishing primordial elements and synthetic elements.
  Fact 0.14: The article has a legend for atomic number colors, where black means solid.

Retrieved items:
<item 1>
Group 3 element
Lanthanum and actinium are sometimes considered the remaining members of group 3 . In their most commonly encountered tripositive ion forms , these elements do
not possess any partially filled f - orbitals , thus continuing the scandium -- yttrium -- lanthanum -- actinium trend , in which all the elements have
relationship similar to that of elements of the calcium -- strontium -- barium -- radium series , the elements ' left neighbors in s - block . However ,
different behavior is observed in other d - block groups , especially in group 4 , in which zirconium , hafnium and rutherfordium share similar chemical
properties lacking a clear trend . It has however been argued that this is irrelevant because the principle of increasing basicity down the table is more
fundamental , and because the behavior of the group 3 elements is more similar to their s - block neighbors than their d - block neighbors . In other tables ,
lutetium and lawrencium are classified as the remaining members of group 3 . In these tables , lutetium and lawrencium end ( or sometimes succeed ) the
lanthanide and actinide series , respectively .
</item 1>

<item 2>
Group 3 element
Group 3 element - wikipedia Group 3 element Jump to : navigation , search For the other group formerly named Group III ( boron group ) , see Group 13 element .
Group 3 in the periodic table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur
Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton
Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium
Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium
Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury ( element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium
</item 2>

<item 3>
Group 3 element
in period 6 and 7 , are in group 3 is disputed . The grouping used in this article places La and Ac in group 3 , which is the most common form . For other
groupings , see group 3 borders . Legend primordial element synthetic element Atomic number color : black = solid Group 3 is a group of elements in the periodic
table . This group , like other d - block groups , should contain four elements , but it is not agreed what elements belong in the group . Scandium ( Sc ) and
yttrium ( Y ) are always included , but the other two spaces are usually occupied by lanthanum ( La ) and actinium ( Ac ) , or by lutetium ( Lu ) and lawrencium
( Lr ) ; less frequently , it is considered the group should be expanded to 32 elements ( with all the lanthanides and actinides included ) or contracted to
contain only scandium and yttrium . When the group is understood to contain all of the lanthanides , its trivial name is the rare - earth metals .
</item 3>

<item 4>
Group 3 element
Characteristics ( edit ) Chemical ( edit ) Electron configurations of the group 3 elements Z Element Electron configuration 21 scandium 2 , 8 , 9 , 2 39 yttrium
2 , 8 , 18 , 9 , 2 57 lanthanum 2 , 8 , 18 , 18 , 9 , 2 89 actinium 2 , 8 , 18 , 32 , 18 , 9 , 2 Like other groups , the members of this family show patterns in
their electron configurations , especially the outermost shells , resulting in trends in chemical behavior . However , lawrencium is an exception , since its
last electron is transferred to the 7p subshell due to relativistic effects . Most of the chemistry has been observed only for the first three members of the
group ; chemical properties of both actinium and especially lawrencium are not well - characterized . The remaining elements of the group ( scandium , yttrium ,
lutetium ) are reactive metals with high melting points ( 1541 ° C , 1526 ° C , 1652 ° C respectively ) .
</item 4>

<item 5>
Group 3 element
element ) Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium
Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium
Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson alkaline earth metals ← → group 4 IUPAC group number Name by element scandium group CAS group
number ( US , pattern A-B-A ) IIIB old IUPAC number ( Europe , pattern A-B ) IIIA ↓ Period Scandium ( Sc ) 21 Transition metal 5 Yttrium ( Y ) 39 Transition
metal 6 Lanthanum ( La * ) 57 Lanthanide 7 Actinium ( Ac * ) 89 Actinide * Whether the elements lutetium ( Lu ) and lawrencium ( Lr ) , in period 6 and 7 , are
in group 3 is disputed .
</item 5>

<item 6>
Group 3 element
Three group 3 elements occur naturally , scandium , yttrium , and either lanthanum or lutetium . Lanthanum continues the trend started by two lighter members in
general chemical behavior , while lutetium behaves more similarly to yttrium . While the choice of lutetium would be in accordance with the trend for period 6
transition metals to behave more similarly to their upper periodic table neighbors , the choice of lanthanum is in accordance with the trends in the s - block ,
which the group 3 elements are chemically more similar to . They all are silvery - white metals under standard conditions . The fourth element , either actinium
or lawrencium , has only radioactive isotopes . Actinium , which occurs only in trace amounts , continues the trend in chemical behavior for metals that form
tripositive ions with a noble gas configuration ; synthetic lawrencium is calculated and partially shown to be more similar to lutetium and yttrium . So far ,
no experiments have been conducted to synthesize any element that could be the next group 3 element .
</item 6>

<item 7>
Group 3 element
^ Jump up to : Barbalace , Kenneth . `` Periodic Table of Elements '' . Environmental Chemistry.com . Retrieved 2007 - 04 - 14 . Jump up ^ `` WebElements
Periodic Table of the Elements '' . Webelements.com . Retrieved 2010 - 04 - 03 . Jump up ^ `` Periodic Table of the Elements '' . International Union of Pure
and Applied Chemistry ( IUPAC ) . Retrieved 2010 - 04 - 03 . ^ Jump up to : `` Visual Elements '' . Royal Society of Chemistry . Retrieved 4 July 2011 . Jump up
^ Dolg , Michael . `` Lanthanides and Actinides '' ( PDF ) . Max - Planck - Institut für Physik komplexer Systeme , Dresden , Germany . CLA01 . Retrieved 4 July
2011 . Jump up ^ IUPAC ( 2015 ) . `` The constitution of group 3 of the periodic table '' . Retrieved 2016 - 11 - 10 . Jump up ^ Greenwood , N.N.
</item 7>

<item 8>
Group 3 element
Exclusion of all elements is based on properties of earlier actinides , which show a much wider variety of chemistry ( for instance , in range of oxidation
states ) within their series than the lanthanides , and comparisons to scandium and yttrium are even less useful . However , these elements are destabilized ,
and if they were stabilized to more closely match chemistry laws , they would be similar to lanthanides as well . Also , the later actinides from californium
onwards behave more like the corresponding lanthanides , with only the valence + 3 ( and sometimes + 2 ) shown . In 2015 , IUPAC initiated a project to decide
... group 3 of the periodic table as consisting either of the elements Sc , Y , Lu and Lr , or the elements Sc , Y , La and Ac . -- IUPAC , The project is led
by Eric Scerri . Apparently , group 3 comprising 32 elements ( as older periodic tables show ) is not considered . The La and Ac variant remains the most common
in the literature , despite some calls for a change to the Lu and Lr variant .
</item 8>

<item 9>
Group 3 element
ISBN 0 - 19 - 850341 - 5 . Links to related articles ( edit ) ( hide ) Periodic table ( Large cells ) 5 6 7 8 9 10 11 12 13 14 15 16 17 18 He Li Be O Ne Na Mg
Al Si Cl Ar Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr 5 Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe 6 Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm
Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn 7 Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og Alkali metal
Alkaline earth metal Lanthanide Actinide Transition metal Post - ​ transition metal Metalloid Reactive nonmetal Noble gas Unknown chemical properties Periodic
table Periodic table forms Standard 18 - column 18 - column ,
</item 9>

<item 10>
Group 3 element
If lanthanum and actinium are considered group 3 elements , then element 121 , unbiunium ( Ubu ) , should be the fifth group 3 element . The element is
calculated have electronic configuration of ( Og ) 8s 8p , which is not associated with transition metals , without having a partially filled d - subshell . No
experiments have been performed to create unpenttrium , unbiunium or any element that could be considered the next group 3 element ; however , unbiunium is the
element with the lowest atomic number that has not been tried to be created and thus has chances to be , while unpenttrium , unpentpentium or any other element
considered if preceded by lawrencium is very unlikely to be created due to drip instabilities that imply that the periodic table ends soon after the island of
stability at unbihexium .
</item 10>
=== END ITEM id=b2a934612f919643 ===

=== ITEM id=d813c3c4d0ec3245 ===
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
The 1979 Conservative Budget of Geoffrey Howe cut the small companies ' rate to 40 % , followed by a further cut in the 1982 Budget to 38 % . The Budgets of
1983 -- 1988 saw sharp cuts in both main and small companies ' rates , falling to 35 % and 25 % respectively . Budgets between 1988 and 2001 brought further
falls to a 30 % main rate and 19 % small companies ' rates . From April 1983 to March 1997 the small companies ' rate was pegged to the basic rate of income tax
. During the 1980s there was briefly a higher rate of tax imposed for capital profits . Chancellor Gordon Brown 's 1999 Budget introduced a 10 % starting rate
for profits from £ 0 to £ 10,000 , effective from April 2000 . Marginal relief applied meaning companies with profits of between £ 10,000 and £ 50,000 paid a
rate between the starting rate and the small companies ' rate ( 19 % in 2000 ) .
</item 1>

<item 2>
United Kingdom corporation tax
The 2002 Budget cut the starting rate to zero , with marginal relief applying in the same way . This caused a significant increase in the number of companies
being incorporated , as businesses that had operated as self - employed , paying income tax on profits from just over £ 5,000 , were attracted to the
corporation tax rate of 0 % on income up to £ 10,000 . Previously self - employed individuals could now distribute profits as dividend payments rather than
salaries . For companies with profits under £ 50,000 the corporation tax rate varied between 0 % and 19 % . Because dividend payments come with a basic rate tax
credit , provided the recipient did not earn more than the basic rate allowance , no further tax would be paid . The number of new companies being formed in
2002 -- 2003 reached 325,900 , an increase of 45 % on 2001 -- 2002 . The fact that individuals operating in this manner could potentially pay no tax at all was
felt by the government to be unfair tax avoidance , and the 2004 Budget introduced a Non-Corporate Distribution Rate .
</item 2>

<item 3>
Graph query row:
entity: United Kingdom corporation tax
relationship_type: LOCATED_IN
relationship: None
relation: None
related_entity: United Kingdom
fact: None
context: None
</item 3>

<item 4>
United Kingdom corporation tax
Financial year FY17 started on 1 April 2017 and ends on 31 March 2018 . Where a company 's accounting period straddles a financial year in which the corporation
tax rate has changed , the company 's profits for that period are split . For example , a company paying small companies ' rate with its accounting period
running from 1 January to 31 December , and making £ 100,000 of profit in 2007 , would be deemed to have made 90 / 365 * £ 100,000 = £ 24,657.53 in FY06 ( there
are 90 days between 1 January and 31 March ) , and 275 / 365 * £ 100,000 = £ 75 , 34.47 in FY07 , and would pay 19 % on the FY06 portion , and 20 % on the FY07
portion . From 1 April 2010 HM Revenue & Customs updated their terminology and the former Small Companies ' Rate is now called Small Profits Rate .
</item 4>

<item 5>
United Kingdom corporation tax
Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 -
2010 to a flat rate of 20 % as of April 2015 . The UK government faced problems with its corporate tax structure , including European Court of Justice
judgements that aspects of it are incompatible with EU treaties . Tax avoidance schemes marketed by the financial sector have also proven an irritant , and been
countered by complicated anti-avoidance legislation . The complexity of the corporation tax system is a recognised issue . The Labour government , supported by
the Opposition parties , carried through wide - scale reform from the Tax Law Rewrite project , resulting in the Corporation Tax Act 2010 . The tax has slowly
been integrating generally accepted accounting practice , with the corporation tax system in various specific areas based directly on the accounting treatment .
Total net corporation tax receipts in 2014 - 15 were £ 43.0 billion , which was an increase of 7 % from £ 40.3 billion in 2013 - 14 .
</item 5>

<item 6>
United Kingdom corporation tax
In March 2001 , the government published a technical note A Review of Small Business Taxation , which considered simplification of corporation tax for small
companies through the closer alignment of their profits for tax purposes with those reported in their accounts . In July of that year , the government also
published a consultation document , Large Business Taxation : the Government 's strategy and corporate tax reforms . It set out the strategy for modernising
corporate taxes and proposals for relief for capital gains on substantial shareholdings held by companies . In August 2002 , Reform of corporation tax -- A
consultation document was published , outlining initial proposals for the abolition of the Schedular system . This was followed up in August 2003 by Corporation
tax reform -- A consultation document , which further discussed the possible abolition of the Schedular system , and also whether the capital allowances ( tax
depreciation ) system should be abolished . It also made proposals that were ultimately enacted in Finance Act 2004 . ( The first two of these listed below were
in response to threats to the UK tax base arising from recent European Court of Justice judgments . ) The changes were to : Introduction of transfer pricing
rules for UK - to - UK transactions .
</item 6>

<item 7>
United Kingdom corporation tax
This ensured that where a company paid below the small companies ' rate ( 19 % in 2004 ) , dividend payments made to non-corporates ( for example , individuals
, trusts and personal representatives of deceased persons ) would be subject to additional corporation tax , bringing the corporation tax paid up to 19 % . For
example , a company making £ 10,000 profit , and making a £ 6,000 dividend distribution to an individual and £ 4,000 to another company would pay 19 %
corporation tax on the £ 6,000 . Although this measure substantially reduced the number of small businesses incorporating , the Chancellor in the 2006 Budget
said tax avoidance by small businesses through incorporation was still a major issue , and scrapped the starting rate entirely . Historic tax revenues ( edit )
UK Corporation Tax Revenue 1999 to 2016 10,000 20,000 30,000 40,000 50,000 Taxable profits and accounting profits ( edit ) The starting point for computing
taxable profits is profits before tax ( except for a life assurance company ) .
</item 7>

<item 8>
United Kingdom corporation tax
These 4.7 % of active companies are responsible for 75 % of all corporation tax receipts . Around 224,000 companies paid the small companies rate , with 34,000
benefiting from marginal relief . 264,000 were in the starting rate , with 269,000 benefiting from the lower band of marginal relief . The total revenue was £
41.9 bn from 831,885 companies . Only 23480 companies had a liability in excess of £ 100,000 . HM Revenue and Customs powers of enquiry ( edit ) HMRC has one
year from the normal filing date , which is itself one year after the end of the period of account , to open an enquiry into the return . This period is
extended if the return is filed late . The enquiry continues until all issues that HMRC wish to enquire about a return are dealt with . However , a company can
appeal to the Commissioners of Income Tax to close an enquiry if they feel there is undue delay . If either side disputes the amount of tax that is payable ,
they may appeal to either the General or Special Commissioners of Income Tax .
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
Corporate Tax rates for Financial Years 2008 -- 2014 2008 -- 2010 2011 2012 2013 2014 Small profits rate 21 % 20 % 20 % 20 % 20 % Small profits upper limit £
300,000 £ 300,000 £ 300,000 £ 300,000 £ 300,000 Marginal relief limits £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £
1,500,000 £ 300,001 -- £ 1,500,000 Main rate 28 % 26 % 24 % 23 % 21 % Notes : The bands shown on the right hand side are divided by one plus the number of
associates ( usually the only associates a company has are fellow group companies , but the term is more widely defined ) The reduced rates do not apply to
close investment holding companies ( companies controlled by fewer than 5 people ( plus associates ) or by their directors / managers , whose main activity is
the holding of investments ) .
</item 10>
=== END ITEM id=d813c3c4d0ec3245 ===

=== ITEM id=d94a410ec2e155b1 ===
Gold sentences and facts:
Sentence 0: Angular cheilitis can be caused by infection , irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may be helped by testing for infections and patch testing for allergies .
  Fact 0.0: Angular cheilitis can be caused by infection.
  Fact 0.1: Angular cheilitis can be caused by irritation.
  Fact 0.2: Angular cheilitis can be caused by allergies.
  Fact 0.3: Infections that cause angular cheilitis include infection by fungi such as Candida albicans.
  Fact 0.4: Infections that cause angular cheilitis include infection by bacteria such as Staph. aureus.
  Fact 0.5: Irritants that cause angular cheilitis include poorly fitting dentures.
  Fact 0.6: Irritants that cause angular cheilitis include licking the lips or drooling.
  Fact 0.7: Irritants that cause angular cheilitis include mouth breathing resulting in a dry mouth.
  Fact 0.8: Irritants that cause angular cheilitis include sun exposure.
  Fact 0.9: Irritants that cause angular cheilitis include overclosure of the mouth.
  Fact 0.10: Irritants that cause angular cheilitis include smoking.
  Fact 0.11: Irritants that cause angular cheilitis include minor trauma.
  Fact 0.12: Allergies that cause angular cheilitis may include substances like toothpaste.
  Fact 0.13: Allergies that cause angular cheilitis may include substances like makeup.
  Fact 0.14: Allergies that cause angular cheilitis may include substances like food.
  Fact 0.15: Often a number of factors are involved in angular cheilitis.
  Fact 0.16: Other factors may include poor nutrition.
  Fact 0.17: Other factors may include poor immune function.
  Fact 0.18: Diagnosis of angular cheilitis may be helped by testing for infections.
  Fact 0.19: Diagnosis of angular cheilitis may be helped by patch testing for allergies.

Retrieved items:
<item 1>
Angular cheilitis, also called angular cheilosis, perleche, or rhagades, involves fissures at the corners of the lips and is well known among denture wearers.
It is classified as a lip and oral-mucosal disorder (ICD-10 K13.0; ICD-9-CM 528.5, 686.8). The source includes references on local causes and dermatology, plus
links to related oral and maxillofacial conditions
</item 1>

<item 2>
Angular cheilitis causes fissured, red, swollen, sometimes crusted or ulcerated sores at mouth corners; it may spread onto facial skin. Symptoms include
soreness, itching, burning, or pain. Chronic cases can involve scaling, pus, and granulation tissue. Causes are multifactorial, often involving yeast or
bacteria, saliva irritation, nutritional deficiency, or poorly fitting dentures and jaw overclosure. Staphylococcus may cause golden crusts. Denture care,
correcting
</item 2>

<item 3>
Angular cheilitis is inflammation at the mouth corners, often caused by saliva pooling in skin folds, leading to moisture damage and yeast infection. It is more
common in people without teeth and denture wearers; reduced facial height from tooth loss or worn dentures can deepen folds. Other contributors include lip
licking, drooling, mouth breathing, dry mouth, eczema, and altered oral anatomy. It may also accompany Down syndrome, inflammatory bowel disease, diabetes, HIV,
immune suppression, or rare
</item 3>

<item 4>
Angular cheilitis
In some cases , the lesion may be confined to the mucosa of the lips , and in other cases the lesion may extend past the vermilion border ( the edge where the
lining on the lips becomes the skin on the face ) onto the facial skin . Initially , the corners of the mouth develop a gray - white thickening and adjacent
erythema ( redness ) . Later , the usual appearance is a roughly triangular area of erythema , edema ( swelling ) and breakdown of skin at either corner of the
mouth . The mucosa of the lip may become fissured ( cracked ) , crusted , ulcerated or atrophied . There is not usually any bleeding . Where the skin is
involved , there may be radiating rhagades ( linear fissures ) from the corner of the mouth . Infrequently , the dermatitis ( which may resemble eczema ) can
extend from the corner of the mouth to the skin of the cheek or chin . If Staphylococcus aureus is involved , the lesion may show golden yellow crusts .
</item 4>

<item 5>
Angular cheilitis is inflammation and cracking at the mouth corners, commonly involving Candida (often with Staphylococcus aureus; mixed infections are
frequent). Causes may include saliva, lip licking, dentures, dry mouth, smoking, trauma, irritants or allergies; iron/B-vitamin deficiencies and impaired
immunity can contribute. Diagnosis may involve infection tests or allergy patch testing. Treatment addresses underlying causes, uses barrier cream, and often
topical antifungal or antibacterial medication. It affects roughly
</item 5>

<item 6>
Angular cheilitis
The corner of the mouth is normally exposed to saliva more than any other part of the lips . Reduced lower facial height ( vertical dimension or facial support
) is usually caused by edentulism ( tooth loss ) , or wearing worn down , old dentures or ones which are not designed optimally . This results in overclosure of
the mandible ( collapse of the jaws ) , which extenuates the angular skin folds at the corners of the mouth , in effect creating an intertriginous skin crease .
The tendency of saliva to pool in these areas is increased , constantly wetting the area , which may cause tissue maceration and favors the development of a
yeast infection . As such , angular cheilitis is more commonly seen in edentulous people ( people without any teeth ) . It is by contrast uncommon in persons
who retain their natural teeth . Angular cheilitis is also commonly seen in denture wearers . Angular cheilitis is present in about 30 % of people with denture
- related stomatitis .
</item 6>

<item 7>
Angular cheilitis
A cold sore at the corner of the mouth behaves similarly to elsewhere on the lips , and follows a pattern of vesicle ( blister ) formation followed by rupture
leaving a crusted sore which resolves in about 7 -- 10 days , and recurs in the same spot periodically , especially during periods of stress . Rather than
utilizing antifungal creams , angular herpes simplex is treated in the same way as a cold sore , with topical antiviral drugs such as aciclovir . Irritation
contact dermatitis ( edit ) A famous sketch by Leonardo da Vinci in preparation to depict the face of Judas Iscariot in The Last supper . The subject shows
overclosure of the jaws and loss of facial support around the mouth . Pronounced skin folds extending from the corner of the mouth . Main article : Irritant
contact dermatitis 22 % of cases of angular cheilitis are due to irritants . Saliva contains digestive enzymes , which may have a degree of digestive action on
tissues if they are left in contact . The corner of the mouth is normally exposed to saliva more than any other part of the lips .
</item 7>

<item 8>
Angular cheilitis (angular cheilosis or perleche) causes redness, cracking, soreness, itching, or crusting at one or both mouth corners; it may be acute or
chronic. Causes are often mixed, especially Candida and Staphylococcus, alongside saliva-related irritation, lip licking, dentures, dry mouth, trauma, or
allergies. Iron or B-vitamin deficiencies and impaired immunity may contribute. Diagnosis is usually clinical, with swabs, blood tests,
</item 8>

<item 9>
Angular cheilitis is inflammation of one or both mouth corners, causing redness, fissures, skin breakdown, crusting, itching, or pain; it may last days to
years. Causes include Candida or bacterial infection, irritation, allergies, dentures, and—especially in developing regions—iron or vitamin deficiencies.
Diagnosis is usually clinical; swabs and blood tests may identify infection or deficiencies. Treatment targets the cause and may include barrier cream, treating
oral infection, and correcting denture fit
</item 9>

<item 10>
Angular cheilitis
Such examples include people with anorexia nervosa . Other disorders may cause lip enlargement ( e.g. orofacial granulomatosis ) , which alters the local
anatomy and extenuates the skin folds at the corners of the mouth . More still may be involved because they affect the immune system , allowing normally
harmless organisms like Candida to become pathogenic and cause an infection . Xerostomia ( dry mouth ) is thought to account for about 5 % of cases of AC .
Xerostomia itself has many possible causes , but commonly the cause may be side effects of medications , or conditions such as Sjögren 's syndrome . Conversely
, conditions which cause drooling or sialorrhoea ( excessive salivation ) can cause angular cheilitis by creating a constant wet environment in the corners of
the mouth . About 25 % of people with Down syndrome appear to have AC . This is due to relative macroglossia , an apparently large tongue in a small mouth ,
which may constantly stick out of the mouth causing maceration of the corners of the mouth with saliva .
</item 10>
=== END ITEM id=d94a410ec2e155b1 ===

=== ITEM id=347542493d4c5763 ===
Gold sentences and facts:
Sentence 0: The small intestine or small bowel is the part of the gastrointestinal tract between the stomach and the large intestine , and is where most of the end absorption of food takes place . The small intestine has three distinct regions -- the duodenum , jejunum , and ileum . The duodenum is the shortest part of the small intestine and is where preparation for absorption begins . It also receives bile and pancreatic juice through the pancreatic duct , controlled by the sphincter of Oddi . The primary function of the small intestine is the absorption of nutrients and minerals from food , using small finger - like protrusions called villi .
  Fact 0.0: The small intestine is also called the small bowel.
  Fact 0.1: The small intestine is the part of the gastrointestinal tract between the stomach and the large intestine.
  Fact 0.2: Most of the end absorption of food takes place in the small intestine.
  Fact 0.3: The small intestine has three distinct regions.
  Fact 0.4: The three regions of the small intestine are the duodenum, jejunum, and ileum.
  Fact 0.5: The duodenum is the shortest part of the small intestine.
  Fact 0.6: Preparation for absorption begins in the duodenum.
  Fact 0.7: The duodenum receives bile through the pancreatic duct.
  Fact 0.8: The duodenum receives pancreatic juice through the pancreatic duct.
  Fact 0.9: The flow of bile and pancreatic juice into the duodenum is controlled by the sphincter of Oddi.
  Fact 0.10: The primary function of the small intestine is the absorption of nutrients and minerals from food.
  Fact 0.11: The small intestine uses small finger-like protrusions called villi for absorption.

Retrieved items:
<item 1>
Human digestive system
Most of the digestion of food takes place in the small intestine . Water and some minerals are reabsorbed back into the blood in the colon of the large
intestine . The waste products of digestion ( feces ) are defecated from the anus via the rectum . Contents ( hide ) 1 Components 1.1 Mouth 1.1. 1 Salivary
glands 1.1. 1.1 Saliva 1.1. 2 Tongue 1.1. 2.1 Taste 1.1. 3 Teeth 1.1. 4 Epiglottis 1.2 Pharynx 1.3 Esophagus 1.4 Diaphragm 1.5 Stomach 1.6 Spleen 1.7 Liver 1.7.
1 Bile 1.7. 2 Gallbladder 1.8 Pancreas 1.9 Lower gastrointestinal tract 1.9. 1 Small intestine 1.9. 2 Cecum 1.9.
</item 1>

<item 2>
Human digestive system
Chyle is then transported through the lymphatic system to the rest of the body . The suspensory muscle marks the end of the duodenum and the division between
the upper gastrointestinal tract and the lower GI tract . The digestive tract continues as the jejunum which continues as the ileum . The jejunum , the
midsection of the small intestine contains circular folds , flaps of doubled mucosal membrane which partially encircle and sometimes completely encircle the
lumen of the intestine . These folds together with villi serve to increase the surface area of the jejunum enabling an increased absorption of digested sugars ,
amino acids and fatty acids into the bloodstream . The circular folds also slow the passage of food giving more time for nutrients to be absorbed . The last
part of the small intestine is the ileum . This also contains villi and vitamin B12 ; bile acids and any residue nutrients are absorbed here . When the chyme is
exhausted of its nutrients the remaining waste material changes into the semi-solids called feces , which pass to the large intestine , where bacteria in the
gut flora further break down residual proteins and starches .
</item 2>

<item 3>
The stomach produces chyme, which passes through the pylorus into the duodenum for further digestion. About 95% of nutrient absorption occurs in the small
intestine; the colon reabsorbs water and minerals and absorbs some bacteria-produced vitamins. Vitamin B12 is protected in the stomach by haptocorrin, then
binds intrinsic factor and is absorbed in the ileum. The stomach can hold about one litre in adults. The spleen breaks down old blood cells; resulting bilirub
</item 3>

<item 4>
Humans cannot digest cellulose because its β-glucose bonds differ from starch’s α-glucose bonds; digestion requires enzymes found in herbivores or some
large-intestine bacteria. The small intestine absorbs most nutrients into blood vessels by diffusion or active transport. Its simple columnar mucosa has
permanent folds (plicae circulares), covered with villi, whose epithelial cells bear microvilli. Rugae, unlike plicae, are temporary folds that allow distention
and contraction.
</item 4>

<item 5>
Small intestine
This is because the cellulose is made out of beta - glucose , making the inter-monosaccharidal bindings different from the ones present in starch , which
consists of alpha - glucose . Humans lack the enzyme for splitting the beta - glucose - bonds , something reserved for herbivores and bacteria from the large
intestine . Absorption ( edit ) Digested food is now able to pass into the blood vessels in the wall of the intestine through either diffusion or active
transport . The small intestine is the site where most of the nutrients from ingested food are absorbed . The inner wall , or mucosa , of the small intestine is
lined with simple columnar epithelial tissue . Structurally , the mucosa is covered in wrinkles or folds called plicae circulares , which are considered
permanent features in the wall of the organ . They are distinct from rugae which are considered non-permanent or temporary allowing for distention and
contraction . From the plicae circulares project microscopic finger - like pieces of tissue called villi ( Latin for `` shaggy hair '' ) . The individual
epithelial cells also have finger - like projections known as microvilli .
</item 5>

<item 6>
The small intestine’s plicae circulares, villi, and epithelial microvilli increase absorptive surface area and reduce nutrient loss to intestinal fauna. Villi
contain capillaries and lymphatic lacteals: epithelial cells send amino acids and carbohydrates into blood capillaries and lipids into lacteals, which transport
them for use by the body. Most nutrients are absorbed in the jejunum; iron mainly in the duodenum, and vitamin B12 and bile salts
</item 6>

<item 7>
The small intestine absorbs water by osmosis, lipids by passive diffusion, sodium bicarbonate by active transport, glucose and amino acids by co-transport, and
fructose by facilitated diffusion. Gut flora and ileal Peyer’s patches support immune defense by sampling antigens. Around 70% of human protein-coding genes are
expressed in the duodenum; examples include FABP2, FABP6, and Paneth-cell DEFA6. Disorders include obstruction (ileus
</item 7>

<item 8>
Human digestive system
This breaks down fats to some degree though is not as efficient as the pancreatic lipase . The pylorus , the lowest section of the stomach which attaches to the
duodenum via the pyloric canal , contains countless glands which secrete digestive enzymes including gastrin . After an hour or two , a thick semi-liquid called
chyme is produced . When the pyloric sphincter , or valve opens , chyme enters the duodenum where it mixes further with digestive enzymes from the pancreas ,
and then passes through the small intestine , where digestion continues . When the chyme is fully digested , it is absorbed into the blood . 95 % of absorption
of nutrients occurs in the small intestine . Water and minerals are reabsorbed back into the blood in the colon of the large intestine , where the environment
is slightly acidic . Some vitamins , such as biotin and vitamin K produced by bacteria in the gut flora of the colon are also absorbed . The parietal cells in
the fundus of the stomach , produce a glycoprotein called intrinsic factor which is essential for the absorption of vitamin B12 .
</item 8>

<item 9>
The small intestine, between the stomach and large intestine, is the main site of nutrient and mineral absorption. Its three regions are the duodenum, jejunum,
and ileum. The duodenum begins digestion and receives bile and pancreatic juice through the pancreatic duct, regulated by the sphincter of Oddi. Villi increase
absorptive surface area. The ileum contains Peyer’s patches; the duodenum has Brunner’s glands. It develops from the embryonic midg
</item 9>

<item 10>
Mucus lubricates the intestine, while small, digested food particles are absorbed through its wall into blood. Chyme enters the duodenal bulb, then passes
through the C-shaped duodenum, where most digestion occurs. Segmentation mixes and slowly moves chyme, allowing absorption. The jejunum’s circular folds and
villi increase surface area for nutrient uptake. The ileum absorbs vitamin B12, bile acids, and remaining nutrients; chyle travels through the lymphatic system.
Remaining
</item 10>
=== END ITEM id=347542493d4c5763 ===

=== ITEM id=473db34da54bb6ce ===
Gold sentences and facts:
Sentence 0: `` Lord Have Mercy on the Working Man '' is a song written by Kostas and recorded by American country music singer Travis Tritt . It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 .
  Fact 0.0: Lord Have Mercy on the Working Man is a song.
  Fact 0.1: Kostas wrote the song Lord Have Mercy on the Working Man.
  Fact 0.2: Travis Tritt, an American country music singer, recorded the song Lord Have Mercy on the Working Man.
  Fact 0.3: Lord Have Mercy on the Working Man was released in August 1992.
  Fact 0.4: Lord Have Mercy on the Working Man was the first of five singles from Travis Tritt's third studio album T-R-O-U-B-L-E.
  Fact 0.5: Lord Have Mercy on the Working Man became Travis Tritt's tenth entry on the Billboard Hot Country Singles & Tracks charts.
  Fact 0.6: The Billboard Hot Country Singles & Tracks chart is now called Hot Country Songs.
  Fact 0.7: Lord Have Mercy on the Working Man peaked at number 5 on the Billboard chart.
Sentence 1: `` Lord Have Mercy on the Working Man '' Single by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology `` Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' ( 1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 )
  Fact 1.0: Lord Have Mercy on the Working Man is a single by Travis Tritt from the album T-R-O-U-B-L-E.
  Fact 1.1: The B-side of the single is Lord Have Mercy on the Working Man (album version).
  Fact 1.2: The single was released on August 10, 1992.
  Fact 1.3: The single format is CD single.
  Fact 1.4: The genre of the single is country.
  Fact 1.5: The album version of the song has a length of 4:55.
  Fact 1.6: The label of the single is Warner Bros. Nashville.
  Fact 1.7: Kostas is the songwriter of Lord Have Mercy on the Working Man.
  Fact 1.8: Gregg Brown is the producer of Lord Have Mercy on the Working Man.
  Fact 1.9: The Travis Tritt singles chronology lists Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).
  Fact 1.10: The Travis Tritt singles chronology then repeats Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).

Retrieved items:
<item 1>
Lord Have Mercy on the Working Man
It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard
Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 . Contents ( hide ) 1 Content 2 Critical reception 3 Music video 4
Chart performance 5 References 6 External links Content ( edit ) `` Lord Have Mercy on the Working Man '' is a moderate up - tempo whose lyrics centralize on a
theme of economic injustice towards blue collar workers . The instrumentation features various forms of percussion from Sam Bacco , including crotales , wobble
board , spoons and a broom . Richard Bennett and Wendell Cox play guitar solos before the third verse , and Brooks & Dunn , T. Graham Brown , George Jones ,
Little Texas , Dana McVicker , Tanya Tucker and Porter Wagoner all sing background vocals on the final chorus .
</item 1>

<item 2>
Lord Have Mercy on the Working Man
Chicago Tribune . Retrieved 28 August 2009 . ^ Jump up to : Whitburn , Joel ( 2006 ) . The Billboard Book of Top 40 Country Hits ( 2 ed . ) . Billboard Books .
p. 353 . ISBN 0 - 8230 - 8291 - 1 . Jump up ^ Walker , Teresa M. ( 4 September 1992 ) . `` Travis Tritt 's rockin ' country '' . Gainesville Sun . Archived from
the original on 12 July 2012 . Retrieved 28 August 2009 . Jump up ^ Larsen , Dave ( 18 September 1992 ) . `` Recordings on review '' . Dayton Daily News .
Retrieved 28 August 2009 . Jump up ^ Nash , Alanna ( 4 September 1992 ) . `` T-R-O-U-B-L-E in mind '' . Entertainment Weekly . Retrieved 28 August 2009 . Jump
up ^ Billboard , August 15 , 1992 Jump up ^ Ellison , Curtis W.
</item 2>

<item 3>
“Lord Have Mercy on the Working Man” is a Travis Tritt song discussed in country-music reviews and chart references, including Billboard, RPM, and Joel
Whitburn’s *Billboard Book of Top 40 Country Hits*. The supplied excerpt lists related 1992 reviews and places the song among singles from Tritt’s
*T-R-O-U-B-L-E* era, alongside other releases in his discography.
</item 3>

<item 4>
Lord Have Mercy on the Working Man
Music video ( edit ) Jack Cole directed the song 's music video . In the book Country Music Culture : From Hard Times to Heaven , Curtis W. Ellison wrote that
the song 's music video `` confronted a litany of personal oppressions attributed to government policy '' that coincided with Bill Clinton 's presidential
campaign . Chart performance ( edit ) `` Lord Have Mercy on the Working Man '' spent 20 weeks on the Billboard Hot Country Singles & Tracks ( now Hot Country
Songs ) charts in 1992 , peaking at number 5 . The song also reached number 10 on the Canadian RPM Country Tracks charts . Chart ( 1992 ) Peak position Canada
Country Tracks ( RPM ) 10 US Hot Country Songs ( Billboard ) 5 References ( edit ) ^ Jump up to : T-R-O-U-B-L-E ( cassette insert ) . Travis Tritt . Warner
Bros. Records . 1992 . 45048 . Jump up ^ Hurst , Jack ( 13 August 1992 ) . `` Satellite show Travis Tritt to launch new album from Park West '' . Chicago
Tribune . Retrieved 28 August 2009 .
</item 4>

<item 5>
“Lord Have Mercy on the Working Man,” released in August 1992 as the first single from Travis Tritt’s album *T-R-O-U-B-L-E*, addresses economic injustice facing
blue-collar workers. It features inventive percussion, guitar solos, and a star-studded backing chorus. Jack Cole directed its music video, which portrayed
personal oppression linked to government policy. Reviews were mixed, though some praised its populist message. The song peaked at No. 5 on the U.S. country
</item 5>

<item 6>
Lord Have Mercy on the Working Man
Lord Have Mercy on the Working Man - wikipedia Lord Have Mercy on the Working Man Jump to : navigation , search `` Lord Have Mercy on the Working Man '' Single
by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single
Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology ``
Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' (
1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Lord Have Mercy on the Working Man '' is a song
written by Kostas and recorded by American country music singer Travis Tritt .
</item 6>

<item 7>
Lord Have Mercy on the Working Man
Critical reception ( edit ) Teresa M. Walker , in her review for the Gainesville Sun , said that with the assistance from superstars on the final chorus , the
song `` should shoot up the charts . '' Dave Larsen of the Dayton Daily News cited it as one of the stronger tracks on the album , saying that the album ``
works best when Tritt sticks with the populist approach . '' Alanna Nash of Entertainment Weekly said of the song , `` Tritt finds a solid image for his laborer
's lament ( ... ) But the melody is so slight that he resorts to an acoustic arrangement that evokes Jimmie Rodgers and the young Roy Acuff . '' Deborah Evans
Price , of Billboard magazine gave the song a mixed review , saying that while the final chorus of song features famous country artists , it was too bad that
the entire song is n't as strong as the last verse . Music video ( edit ) Jack Cole directed the song 's music video .
</item 7>

<item 8>
“Lord Have Mercy on the Working Man” is a country song written by Kostas and recorded by Travis Tritt. Released as a Warner Bros. Nashville CD single on August
10, 1992, it appeared on Tritt’s album *T-R-O-U-B-L-E*. Gregg Brown produced the 4:55 album version. It followed “Nothing Short of Dying” and preceded “Can I
Trust You with My Heart” in Tritt’s singles chronology.
</item 8>

<item 9>
Lord Have Mercy on the Working Man
( 1995 ) . Country Music Culture : From Hard Times to Heaven . Univ . Press of Mississippi . p. 233 . ISBN 0 - 87805 - 722 - 6 . Retrieved 28 August 2009 . Jump
up ^ `` Top RPM Country Tracks : Issue 1864 . '' RPM . Library and Archives Canada . November 28 , 1992 . Retrieved August 15 , 2013 . Jump up ^ `` Travis Tritt
Chart History ( Hot Country Songs ) '' . Billboard . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Travis Tritt singles Country Club ``
Country Club '' `` Help Me Hold On '' `` I 'm Gonna Be Somebody '' `` Put Some Drive in Your Country '' `` Drift Off to Dream '' It 's All About to Change ``
Here 's a Quarter ( Call Someone Who Cares ) '' `` Anymore '' `` The Whiskey Ai n't Workin ' '' ( With Marty Stuart ) `` Nothing Short of Dying '' T-R-O-U-B-L-E
`` Lord Have Mercy on the Working Man '' ``
</item 9>

<item 10>
Lord Have Mercy on the Working Man
On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lord_Have_Mercy_on_the_Working_Man&oldid=789364103 '' Categories : 1992 singles
Travis Tritt songs Songs written by Kostas ( songwriter ) Warner Bros. Records singles 1992 songs Hidden categories : Articles with hAudio microformats
Singlechart usages for Canadacountry Singlechart called without artist Singlechart called without song Singlechart usages for Billboardcountrysongs Talk
Contents About Wikipedia Add links This page was last edited on 6 July 2017 , at 22 : 41 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=473db34da54bb6ce ===

=== ITEM id=64a58ed139d49a9b ===
Gold sentences and facts:
Sentence 0: `` Too Much Time on My Hands '' is the second single released from Styx 's 1981 triple - platinum album Paradise Theatre . It was written and sung by Tommy Shaw , who also plays the lead guitar solo during the break in the song .
  Fact 0.0: 'Too Much Time on My Hands' is the second single released from Styx's 1981 triple-platinum album Paradise Theatre.
  Fact 0.1: 'Too Much Time on My Hands' was written by Tommy Shaw.
  Fact 0.2: 'Too Much Time on My Hands' was sung by Tommy Shaw.
  Fact 0.3: Tommy Shaw plays the lead guitar solo during the break in 'Too Much Time on My Hands'.

Retrieved items:
<item 1>
Too Much Time on My Hands
Louis ( 2002 ) 21st Century Live ( 2003 ) One with Everything : Styx and the Contemporary Youth Orchestra ( 2006 ) The Grand Illusion / Pieces of Eight - Live (
2012 ) Live at the Orleans Arena , Las Vegas ( 2015 ) Compilations Best of Styx ( 1977 ) Lady ( 1980 ) Styx Classics Volume 15 ( 1987 ) Styx Radio - Made Hits
1975 - 1991 ( 1991 ) Greatest Hits ( 1992 ) Greatest Hits ( 1995 ) Styx Greatest Hits Part 2 ( 1996 ) The Best of Times : The Best of Styx ( 1997 ) Best of Styx
1973 - 1974 ( 1999 ) Extended Versions ( 2000 ) Singles Collection ( 2000 ) Styx Yesterday & Today ( 2001 ) 20th Century Masters ( 2002 ) Rockers ( 2003 ) Come
Sail Away -- The Styx Anthology ( 2004 ) The Complete Wooden Nickel Recordings ( 2005 ) Regeneration : Volume I & II ( 2011 & 2012 ) Cover albums
</item 1>

<item 2>
Closing Time (album)
`` Closing Time '' ( instrumental ) 4 : 20 Total length : 45 : 46 Personnel ( edit ) Musicians ( edit ) Tom Waits -- vocals , piano , guitar , harmonium ,
harpsichord , celeste Delbert Bennett -- trumpet Shep Cooke -- guitar , backing vocals on `` Old Shoes ( & Picture Postcards ) '' Peter Klimes -- guitar , pedal
steel guitar on `` Rosie '' Bill Plummer -- double bass John Seiter -- drums ; backing vocals on `` Ol ' 55 '' and `` Rosie '' Guest musicians ( edit ) Arni
Egilsson -- bass guitar on `` Closing Time '' ( Instrumental ) Jesse Ehrlich -- cello on `` Martha '' Tony Terran -- trumpet on `` Closing Time '' (
Instrumental ) Technical personnel ( edit ) Ed Caraeff -- photography Richie Moore -- additional engineering Cal Schenkel -- design , artwork Jerry Yester --
production , engineering Chart positions ( edit ) Chart ( 2000 ) Peak position Irish Albums Chart 29 Certifications ( edit ) Region Certification Certified
units / Sales United Kingdom ( BPI ) Gold 100,000 Summaries shipments figures based on certification alone References ( edit ) Jump up ^ `` Releases '' .
</item 2>

<item 3>
Too Much Time on My Hands
Too Much Time on My Hands - Wikipedia Too Much Time on My Hands Jump to : navigation , search `` Too Much Time on My Hands '' Single by Styx from the album
Paradise Theatre B - side `` Queen of Spades '' Released March 1981 Format 7 '' Recorded 1980 Length 4 : 31 Label A&M Songwriter ( s ) Tommy Shaw Producer ( s )
Styx Styx singles chronology `` The Best of Times '' ( 1981 ) `` Too Much Time on My Hands '' ( 1981 ) `` Nothing Ever Goes as Planned '' ( 1981 ) `` The Best
of Times '' ( 1981 ) `` Too Much Time on My Hands '' ( 1981 ) `` Nothing Ever Goes as Planned '' ( 1981 ) `` Too Much Time on My Hands '' is the second single
released from Styx 's 1981 triple - platinum album Paradise Theatre . It was written and sung by Tommy Shaw , who also plays the lead guitar solo during the
break in the song . It reached No. 9 on the US Billboard Hot 100 , No. 2 on the Top Rock Tracks chart , and No.
</item 3>

<item 4>
Too Much Time on My Hands
) Regeneration : Volume I & II ( 2011 & 2012 ) Cover albums Big Bang Theory ( 2005 ) Songs `` Best Thing '' ( 1972 ) `` Lady '' ( 1973 ) `` You Need Love '' (
1975 ) `` Lorelei '' ( 1976 ) `` Mademoiselle '' ( 1976 ) `` Crystal Ball '' ( 1977 ) `` Come Sail Away '' ( 1977 ) `` Fooling Yourself ( The Angry Young Man )
'' ( 1978 ) `` Blue Collar Man ( Long Nights ) '' ( 1978 ) `` Sing for the Day '' ( 1979 ) `` Renegade ( 1979 ) `` Babe '' ( 1979 ) `` Why Me '' ( 1980 ) ``
Boat on the River ( 1980 ) `` Borrowed Time '' ( 1980 ) `` Lights '' ( 1980 ) `` First Time '' ( 1980 ) `` The Best of Times '' ( 1981 ) `` Too Much Time on My
Hands '' ( 1981 ) `` Nothing Ever Goes as Planned '' ( 1981 ) `` Rockin ' the Paradise '' ( 1981 ) ``
</item 4>

<item 5>
Too Much Time on My Hands
Planned '' ( 1981 ) `` Rockin ' the Paradise '' ( 1981 ) `` Snowblind '' ( 1981 ) `` Mr. Roboto '' ( 1983 ) `` Do n't Let It End '' ( 1983 ) `` High Time '' (
1983 ) `` Music Time '' / `` Heavy Metal Poisoning '' ( 1984 ) `` Love Is the Ritual '' ( 1990 ) `` Show Me the Way '' ( 1991 ) `` Love at First Sight '' ( 1991
) `` Paradise '' ( 1997 ) `` On My Way '' ( 1997 ) `` Waiting for Our Time '' ( 2003 ) `` I Am the Walrus '' ( 2003 ) `` Ca n't Stop Rockin ' '' ( 2009 ) ``
Gone Gone Gone '' ( 2017 ) Related articles Discography Book This 1980s rock song - related article is a stub . You can help Wikipedia by expanding it .
</item 5>

<item 6>
Too Much Time on My Hands
2 on the Top Rock Tracks chart , and No. 4 on the RPM Top Singles chart of Canada . Contents ( hide ) 1 Personnel 2 In popular culture 3 References 4 External
links Personnel ( edit ) Tommy Shaw -- lead vocals , lead guitar Dennis DeYoung -- keyboards , backing vocals James Young -- rhythm guitar , backing vocals
Chuck Panozzo -- bass John Panozzo -- drums In popular culture ( edit ) The song enjoyed a small revival when The Tonight Show host Jimmy Fallon started talking
about the music video in April 2016 . This culminated with the episode on April 29 , 2016 when Fallon showed a shot - by - shot reenactment of the video with
him and guest Paul Rudd on the show . In this video Fallon plays the part of Dennis DeYoung and Rudd plays Tommy Shaw , with A.D. Miles as James Young , Seth
Herzog as John Panozzo , and Gerard Bradford as Chuck Panozzo . Tommy Shaw has commented positively and said he was impressed with Fallon 's vocals . References
( edit ) Jump up ^ `` Styx '' .
</item 6>

<item 7>
Graph query row:
song: Too Much Time on My Hands
relationship: RELATED_TO
connected_entity: Tommy Shaw
</item 7>

<item 8>
Too Much Time on My Hands
Styx . External links ( edit ) Songfacts ( hide ) Styx Chuck Panozzo James `` J.Y. '' Young Tommy Shaw Todd Sucherman Lawrence Gowan Ricky Phillips Dennis
DeYoung John Panozzo John `` J.C. '' Curulewski Glen Burtnik Studio albums Styx ( 1972 ) Styx II ( 1973 ) The Serpent Is Rising ( 1973 ) Man of Miracles ( 1974
) Equinox ( 1975 ) Crystal Ball ( 1976 ) The Grand Illusion ( 1977 ) Pieces of Eight ( 1978 ) Cornerstone ( 1979 ) Paradise Theatre ( 1981 ) Kilroy Was Here (
1983 ) Edge of the Century ( 1990 ) Brave New World ( 1999 ) Cyclorama ( 2003 ) The Mission ( 2017 ) Live albums Caught in the Act ( 1984 ) Return to Paradise (
1997 ) Arch Allies : Live at Riverport ( 2000 ) Styx World : Live 2001 ( 2001 ) At the River 's Edge : Live in St.
</item 8>

<item 9>
Too Much Time on My Hands
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Too_Much_Time_on_My_Hands&oldid=836493612 '' Categories :
1981 singles Styx ( band ) songs Songs written by Tommy Shaw A&M Records singles 1980 songs 1980s rock song stubs Hidden categories : Articles with hAudio
microformats All stub articles Talk Contents About Wikipedia Add links This page was last edited on 15 April 2018 , at 03 : 30 . About Wikipedia
</item 9>

<item 10>
Too Much Time on My Hands
References ( edit ) Jump up ^ `` Styx '' . Jump up ^ `` Jimmy Fallon & Paul Rudd recreate Styx Too Much Time On My Hands video - EW.com '' . Entertainment
Weekly 's EW.com . Jump up ^ `` Jimmy Fallon and Paul Rudd re-create Styx 's ' Too Much Time On My Hands ' video '' . UPI. 30 April 2016 . Jump up ^ Jimmy
Fallon & Paul Rudd Recreate Styx Music Video . 29 April 2016 -- via YouTube . Jump up ^ `` Watch Jimmy Fallon 's and Paul Rudd 's Remake of Styx 's ' Too Much
Time on My Hands ' Video '' . Ultimate Classic Rock . 30 April 2016 . Jump up ^ Danielle Henderson ( 30 April 2016 ) . `` Jimmy Fallon and Paul Rudd Gave Us the
Shot - For - Shot Styx Video Remake We Did n't Know We Needed '' . Esquire . Jump up ^ `` Jimmy Fallon Sings `` Too Much Time on My Hands '' - Styx `` . Styx .
</item 10>
=== END ITEM id=64a58ed139d49a9b ===

=== ITEM id=3effaff7c7ad43bb ===
Gold sentences and facts:
Sentence 0: `` Baby , Now That I 've Found You '' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh , Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' ( Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 ( From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by The Cox Family ) `` I Will '' ( John Lennon , Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album '' performed by Alison Krauss & Union Station )
  Fact 0.0: 'Baby, Now That I've Found You' was written by Tony Macaulay and John MacLeod.
  Fact 0.1: 'Baby, Now That I've Found You' runs 3 minutes 49 seconds.
  Fact 0.2: 'Baby, Now That I've Found You' is previously unreleased.
  Fact 0.3: 'Oh, Atlanta' was written by Mick Ralphs.
  Fact 0.4: 'Oh, Atlanta' runs 4 minutes 41 seconds.
  Fact 0.5: 'Oh, Atlanta' is previously unreleased.
  Fact 0.6: 'Broadway' was written by Sidney & Suzanne Cox.
  Fact 0.7: 'Broadway' runs 3 minutes 57 seconds.
  Fact 0.8: 'Broadway' is previously unreleased.
  Fact 0.9: 'Every Time You Say Goodbye' was written by John Pennell.
  Fact 0.10: 'Every Time You Say Goodbye' runs 3 minutes 13 seconds.
  Fact 0.11: 'Every Time You Say Goodbye' is from the album 'Every Time You Say Goodbye' by Alison Krauss & Union Station.
  Fact 0.12: 'Tonight I'll Be Lonely Too' was written by Sidney Cox.
  Fact 0.13: 'Tonight I'll Be Lonely Too' runs 3 minutes 25 seconds.
  Fact 0.14: 'Tonight I'll Be Lonely Too' is from the album 'I've Got That Old Feeling' by Alison Krauss.
  Fact 0.15: 'Teardrops Will Kiss the Morning Dew' was written by Paul Craft.
  Fact 0.16: 'Teardrops Will Kiss the Morning Dew' runs 3 minutes 38 seconds.
  Fact 0.17: 'Teardrops Will Kiss the Morning Dew' is from the album 'Two Highways' by Alison Krauss & Union Station.
  Fact 0.18: 'Sleep On' was written by Nelson Mandrell.
  Fact 0.19: 'Sleep On' runs 2 minutes 23 seconds.
  Fact 0.20: 'Sleep On' is from the album 'Too Late to Cry' by Alison Krauss.
  Fact 0.21: 'When God Dips His Pen of Love in My Heart' was written by Cleavant Derricks.
  Fact 0.22: 'When God Dips His Pen of Love in My Heart' runs 2 minutes 26 seconds.
  Fact 0.23: 'When God Dips His Pen of Love in My Heart' is from the album 'Everybody's Reaching Out for Someone' by The Cox Family.
  Fact 0.24: 'I Will' was written by John Lennon and Paul McCartney.
  Fact 0.25: 'I Will' runs 4 minutes 4 seconds.
  Fact 0.26: 'I Will' is from the album 'Within Reach' by Tony Furtado.
  Fact 0.27: 'I Don't Believe You've Met My Baby' was written by Autry Inman.
  Fact 0.28: 'I Don't Believe You've Met My Baby' runs 3 minutes 10 seconds.
  Fact 0.29: 'I Don't Believe You've Met My Baby' is from the album 'Slide Rule' by Jerry Douglas.
  Fact 0.30: 'In the Palm of Your Hand' was written by Ron Block.
  Fact 0.31: 'In the Palm of Your Hand' runs 3 minutes 24 seconds.
  Fact 0.32: 'In the Palm of Your Hand' is from the album 'I Know Who Holds Tomorrow' by Alison Krauss & The Cox Family.
  Fact 0.33: 'When You Say Nothing at All' was written by Paul Overstreet and Don Schlitz.
  Fact 0.34: 'When You Say Nothing at All' runs 4 minutes 21 seconds.
  Fact 0.35: 'When You Say Nothing at All' is performed by Alison Krauss & Union Station on the album 'Keith Whitley: A Tribute Album'.

Retrieved items:
<item 1>
Now That I've Found You: A Collection
Brent Truitt : Mandolin Stuart Duncan : Fiddle , Harmonica , Cymbals Adam Steffey : Mandolin & Mandola Pete Wasner : Piano Chart performance ( edit ) Album (
edit ) Chart ( 1995 ) Peak position Canadian RPM Country Albums U.S. Billboard 200 13 U.S. Billboard Top Country Albums References ( edit ) Jump up ^ Allmusic
review Jump up ^ Robert Christgau review Jump up ^ Rolling Stone review ^ Jump up to : Alison Krauss Artist Chart Position at Billboard Now That I 've Found You
for Rounder Records . Retrieved August 28 , 2010 . Alison Krauss bio for CMT.com . Retrieved June 7 , 2006 . Now That I 've Found You on Amazon . Retrieved June
7 , 2006 . Preceded by The Woman in Me by Shania Twain RPM Country Albums number - one album October 30 - November 5 ,
</item 1>

<item 2>
Now That I've Found You: A Collection
It is a retrospective of the early part of Krauss ' recording career . It includes songs that appeared on her solo albums , albums by Alison Krauss & Union
Station , and some that appeared on an album by Alison Krauss & the Cox Family . It also contains some new material . The album features the hit country single
`` When You Say Nothing at All '' , sold over two million copies earning double platinum RIAA certification . The title single `` Baby , Now That I 've Found
You '' won the Grammy Award for Best Female Country Vocal Performance and `` When You Say Nothing at All '' was named Single of the Year by the Country Music
Association . Contents ( hide ) 1 Track listing 2 Personnel 3 Chart performance 3.1 Album 4 References Track listing ( edit ) `` Baby , Now That I 've Found You
'' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh ,
</item 2>

<item 3>
Now That I've Found You: A Collection
Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say
Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' (
Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 (
From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by
Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by
The Cox Family ) `` I Will '' ( John Lennon ,
</item 3>

<item 4>
What Ifs
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
</item 4>

<item 5>
Now That I've Found You: A Collection
1995 Succeeded by Starting Over by Reba McEntire ( hide ) Alison Krauss & Union Station Alison Krauss Barry Bales Ron Block Jerry Douglas Dan Tyminski John
Pennell Jeff White Brent Truitt Mike Harman Adam Steffey Tim Stafford Studio albums Different Strokes Too Late to Cry Two Highways I 've Got That Old Feeling
Every Time You Say Goodbye I Know Who Holds Tomorrow So Long So Wrong Forget About It New Favorite Lonely Runs Both Ways Paper Airplane Windy City Compilation
albums Now That I 've Found You : A Collection Live A Hundred Miles or More : A Collection Essential Alison Krauss With Robert Plant Raising Sand Singles ``
When You Say Nothing at All '' `` Baby Now That I 've Found You '' `` Baby Mine '' `` The Lucky One '' `` Missing You '' `` Please Read the Letter '' `` Rich
Woman '' Featured singles `` Teach Your Children '' `` Somewhere in the Vicinity of the Heart '' `` It 's Not Over '' `` Buy Me a Rose '' `` Whiskey Lullaby ''
Related articles Discography Awards and nominations Alison Brown Robert Plant Viktor Krauss Up !
</item 5>

<item 6>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Now That I've Found You: A Collection
Now That I 've Found You : a Collection - wikipedia Now That I 've Found You : a Collection Jump to : navigation , search Now That I 've Found You : A
Collection Compilation album by Alison Krauss Released February 7 , 1995 Genre Bluegrass / Country Length 42 : 31 Label Rounder Producer Alison Krauss & Union
Station Alison Krauss chronology I Know Who Holds Tomorrow ( 1994 ) I Know Who Holds Tomorrow 1994 Now That I 've Found You : A Collection ( 1995 ) So Long So
Wrong ( 1997 ) So Long So Wrong 1997 Singles from Now That I 've Found You : A Collection `` Baby , Now That I 've Found You '' Released : 1995 `` Oh , Atlanta
'' Released : 1995 Professional ratings Review scores Source Rating Allmusic Robert Christgau A − Rolling Stone Now That I 've Found You : A Collection is an
album by Alison Krauss , released February 7 , 1995 . It is a retrospective of the early part of Krauss ' recording career .
</item 7>

<item 8>
Now That I've Found You: A Collection
Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the
Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison
Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album ''
performed by Alison Krauss & Union Station ) Personnel ( edit ) Alison Krauss : Vocal , Fiddle , Viola Ron Block : Vocal , Acoustic & Electric Guitars , Banjo
Russ Barenberg , Mike Marshall , Scott Nygaard , Tim Stafford : Guitars Jerry Douglas : Guitars , Dobro Sam Bush , Brent Truitt : Mandolin Stuart Duncan :
Fiddle , Harmonica ,
</item 8>

<item 9>
For What It's Worth
I Hope You Find It '' `` Take It Like a Man '' `` I Walk Alone '' Guest singles `` A Love Like Yours ( Do n't Come Knocking Everyday ) '' `` Dead Ringer for
Love '' `` I Got You Babe '' `` It Ai n't Necessarily So '' `` Love Can Build a Bridge '' `` Più che puoi '' `` Bewitched , Bothered and Bewildered '' Book
Portal Haley Reinhart Discography Videography Songs Studio albums Listen Up ! ( 2012 ) Better ( 2016 ) What 's That Sound ? ( 2017 ) EPs American Idol Season 10
Highlights : Haley Reinhart ( 2011 ) Songs `` Baby , It 's Cold Outside '' `` Free '' `` Ca n't Help Falling in Love '' `` Better '' Concert tours American
Idols LIVE !
</item 9>

<item 10>
Now That I've Found You: A Collection
Close and Personal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Now_That_I%27ve_Found_You:_A_Collection&oldid=808197945 '' Categories : 1995
compilation albums Alison Krauss & Union Station albums Rounder Records compilation albums Hidden categories : Articles with hAudio microformats Music infoboxes
with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 1 November 2017 , at 13 : 34 . About Wikipedia
</item 10>
=== END ITEM id=3effaff7c7ad43bb ===
