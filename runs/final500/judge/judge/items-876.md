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

=== ITEM id=5bdfa46a3df716c9 ===
Gold sentences and facts:
Sentence 0: The military convention between Bulgaria and the Central Powers laid down the general plan for its campaign against the Kingdom of Serbia . It severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army , which was designated part of a combined German , Bulgarian and Austro - Hungarian force commanded by Field Marshal August von Mackensen . He had recently led the German and Austro - Hungarian armies in the highly successful and victorious Gorlice -- Tarnów Offensive of the Central Powers against the Russian army on the Eastern Front . His army group was created specifically to wage war against the Serbian army in the pre-1913 borders of the country ( `` Old Serbia '' ) , to defeat it wherever it found it and to open the land route between Hungary and Bulgaria . As commander , Mackensen acted independently and received his directives only from the German High Command . However , the field marshal 's orders to his Bulgarian forces had to be relayed to the commander of the 1st Army by the Bulgarian General Staff , which left room for the latter to intervene when needed . According to the convention , the Bulgarian commander - in - chief retained full and direct control over the Bulgarian 2nd Army and its operations in Vardar Macedonia .
  Fact 0.0: The military convention between Bulgaria and the Central Powers laid down the general plan for Bulgaria's campaign against the Kingdom of Serbia.
  Fact 0.1: The military convention severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army.
  Fact 0.2: The Bulgarian 1st Army was designated part of a combined German, Bulgarian and Austro-Hungarian force.
  Fact 0.3: The combined force was commanded by Field Marshal August von Mackensen.
  Fact 0.4: Mackensen had recently led the German and Austro-Hungarian armies in the highly successful and victorious Gorlice-Tarnów Offensive of the Central Powers against the Russian army on the Eastern Front.
  Fact 0.5: Mackensen's army group was created specifically to wage war against the Serbian army in the pre-1913 borders of the country ('Old Serbia').
  Fact 0.6: Mackensen's army group was created to defeat the Serbian army wherever it found it.
  Fact 0.7: Mackensen's army group was created to open the land route between Hungary and Bulgaria.
  Fact 0.8: As commander, Mackensen acted independently.
  Fact 0.9: Mackensen received his directives only from the German High Command.
  Fact 0.10: The field marshal's orders to his Bulgarian forces had to be relayed to the commander of the 1st Army by the Bulgarian General Staff.
  Fact 0.11: The relay of orders through the Bulgarian General Staff left room for the Bulgarian General Staff to intervene when needed.
  Fact 0.12: According to the convention, the Bulgarian commander-in-chief retained full and direct control over the Bulgarian 2nd Army.
  Fact 0.13: According to the convention, the Bulgarian commander-in-chief retained full and direct control over the 2nd Army's operations in Vardar Macedonia.

Retrieved items:
<item 1>
Bulgaria during World War I
In November , while the decisive Serbian defeat unfolded , the French attempted to exert pressure on the Bulgarian 2nd Army , but were soon forced to halt their
attempts to drive north . The forces of General Maurice Sarrail that consisted of three French and one British division dug in along an 80 - kilometer front
from the river Cherna to Lake Doiran . With the fall of Pristina , Generail Sarrail realized that the Allies could no longer help the Serbians and decided to
begin pulling back his forces to Salonika . The Bulgarian High Command shifted its focus to the Allies in Macedonia and decided that the time was right to go on
the offensive . Several days were lost , however , in scouting , and it was only on 3 December that the 2nd army commenced a general advance . Nonetheless , the
French were able to retreat in good order towards Salonika . They were soon followed by the British , who were defeated at the Battle of Kosturino . On 11
December , the Bulgarian divisions reached the Greek border , where they were ordered to halt and warned repeatedly not to cross .
</item 1>

<item 2>
Bulgaria during World War I
The mood in Saint Petersburg was far less optimistic , as the Russians were aware that the price tag of Romania 's entry into the war would include Bessarabia
and also feared that its intervention would only extend the already huge Eastern Front . The initial reaction of the Central Powers to the declaration of
neutrality was similar to that of the Entente . Germany and Austria - Hungary in particular were weighing the possibility of encouraging immediate Bulgarian
intervention against Serbia , because the neutrality of Italy and Romania , countries that had been their allies prior to the outbreak of the war , was a major
defeat for German and Austrian diplomacy . Radoslavov , who was generally pro-German , engaged in talks with the German and Austro - Hungarian ambassadors as
early as July 1914 , but in the end he preferred to reaffirm Bulgaria 's neutrality . On other diplomatic fronts , the prime minister achieved greater results
with the signing of a secret treaty between Bulgaria and the Ottoman Empire on 6 August 1914 . This was a mutual defense pact that would come into effect if
either party was attacked by another Balkan power . Both countries pledged not to attack other Balkan countries without consultation with each other .
</item 2>

<item 3>
Bulgaria during World War I
Strategic geographic location and a strong military establishment made the country a desired ally for both warring coalitions , but Bulgaria 's regional
aspirations were difficult to satisfy because they included territorial claims against four Balkan countries . As the war progressed , the Central Powers of
Austria - Hungary and the German Empire found themselves in a better position to fulfil Bulgarian demands and persuaded the country to join their cause in
September 1915 . Though the smallest member of the Central Powers in area and in population , Bulgaria made vital contributions to their common war effort . Its
entry to the war heralded the defeat of Serbia , thwarted the foreign - policy goals of Romania , and ensured the continuation of the Ottoman war effort by
providing a geographical conduit for material assistance from Germany to Istanbul . Though the Balkan theatre of the war saw successful campaigns of rapid
movement in 1915 and 1916 , the conflict degraded into a state of attritional trench warfare on both the Northern and the Southern Bulgarian Fronts after most
Bulgarian territorial aspirations had been satisfied . This period of the war substantially weakened the Bulgarian economy , created various supply problems and
reduced the health and morale of Bulgarian troops on the front lines .
</item 3>

<item 4>
Bulgaria during World War I
Russia : Andrei Zayonchkovski Vladimir Sakharov Field Armies Third Army Battles 1916 Battle of Turtucaia Battle of Dobrich First Cobadin Flămânda Offensive
Second Cobadin Battle of Bucharest Outcome 1918 Treaty of Brest - Litovsk Armistice of Focșani Treaty of Bucharest Protocol of Berlin Outcome Armistice of
Salonica Vladaya revolt Treaty of Neuilly - sur - Seine Others Bulgarian administration in Kosovo Anti-military propaganda Ferdinand Vasil Radoslavov Aleksandar
Malinov Aleksandar Stamboliyski Aleksandar Protogerov August von Mackensen Boris Drangov Vladimir Vazov Georgi Todorov ( general ) Ivan Kolev Kliment Boyadzhiev
Dimitar Geshov Ivan Lukov Konstantin Zhostov Krastyu Zlatarev Nikola Zhekov Panteley Kiselov Stefan Toshev Stefan Nerezov Todor Kantardzhiev Retrieved from ``
https://en.wikipedia.org/w/index.
</item 4>

<item 5>
League of Nations
Pacific and South Atlantic Oceans Papeete Madras Penang Cocos Coronel Falkland Islands Más a Tierra Principal participants ( people ) Entente powers Belgium
Brazil China France French Empire Greece Italy Japan Montenegro Portuguese Empire Romania Russia Russian Empire Russian Republic Serbia United Kingdom British
Empire United States Central Powers Germany Austria - Hungary Ottoman Empire Bulgaria Timeline Pre-War conflicts Scramble for Africa ( 1880 -- 1914 ) Russo -
Japanese War ( 1905 ) First Moroccan ( Tangier ) Crisis ( 1905 -- 06 ) Agadir Crisis ( 1911 ) Italo - Turkish War ( 1911 -- 12 ) French conquest of Morocco (
1911 -- 12 ) First Balkan War ( 1912 -- 13 ) Second Balkan War ( 1913 ) Prelude Origins Sarajevo assassination Anti-Serb riots in Sarajevo July Crisis Autumn
1914 Battle of the Frontiers Battle of Cer First Battle of the Marne Siege of Tsingtao Battle of Tannenberg Battle of Galicia Battle of the Masurian Lakes
Battle of Kolubara Battle of Sarikamish Race to the Sea First Battle of Ypres 1915 Second Battle of the Masurian Lakes
</item 5>

<item 6>
Bulgaria during World War I
Allied representatives met with the leaders of the Bulgarian opposition parties and also provided generous financial support for opposition newspapers ; they
even attempted to bribe high - ranking government officials . Germany and Austria - Hungary were not willing to remain on the sidelines and dispatched to
Bulgaria Duke John Albert of Mecklenburg , the former ambassador to the Ottoman Empire Hans Freiherr von Wangenheim and Prince Hohenlohe , who openly declared
that after the defeat of Serbia , Bulgaria would assume hegemony of the Balkans . What seized Bulgarian interest the most was indeed the balance of military
power . The situation on the major European fronts was at that time developing markedly in favor of the Central Powers , and while the Allied operation in
Gallipoli turned into a costly stalemate , the Russians were being driven out of Galicia and Poland . Under these circumstances . the Central Powers were hoping
to secure Bulgaria at last . Still , it took Entente diplomacy more than a month to give an answer to Radoslavov 's questions and the reply proved far from
satisfactory . In reality , it hardly differed from the offer the Allies presented in May .
</item 6>

<item 7>
Bulgaria during World War I
By the middle of December , the entire Kingdom of Serbia was occupied by the armies of the Central Powers and the Allies were pushed back to Salonika by the
Bulgarians . Within two months of its entry into the war Bulgaria , achieved its main war goal : the conquest of Vardar Macedonia . For the duration of military
operations against Serbia and the Entente in 1915 , the Bulgarian Army committed a total of around 424,375 men , while its casualties were held to around 37,000
men . By the end of 1915 , the Central Powers had established firm and unbroken control over a vast territory that stretched from the North Sea to Mesopotamia .
They also drew great political and military dividends from the defeat and occupation of Serbia . Bulgaria conquered almost all the territory it desired ,
Germany gained unrestricted access to the natural resources of Ottoman Asia , the Ottomans received much needed German matériel assistance and Austria - Hungary
secured its southern flank and could completely focus its attention on the Russian and Italian fronts .
</item 7>

<item 8>
Bulgaria during World War I
The end of neutrality ( edit ) As 1914 came to an end , Bulgaria remained on the sidelines of World War I. Popular opinion lacked enthusiasm for entering the
conflict and supported the country 's stance of neutrality . At this point , Prime Minister Radoslavov adopted a `` wait and see '' policy while at the same
time he successfully probed the abilities of the warring alliances to satisfy Bulgarian territorial ambitions . A final commitment could be made only when one
of the sides had gained a decisive military advantage and had firmly guaranteed the fulfillment of Bulgarian national ideals . On the battlefields far from
Bulgaria , the war had entered a long period of stalemate with no side appearing to gain the upper hand . On the Western Front in February 1915 , the French
failed to break the German lines at the First Battle of Champagne , while further attempts at the Second Battle of Artois during May also came to an
unsuccessful conclusion . The Germans had decided to focus their efforts on the Eastern Front , where they had considerable success against the Russians in the
Second Battle of the Masurian Lakes in February 1915 , but their gains were largely negated at the Siege of Przemyśl in March .
</item 8>

<item 9>
League of Nations
Civil War ( 1922 -- 23 ) Aspects Opposition Pacifism Anti-war movement Deployment Schlieffen Plan ( German ) Plan XVII ( French ) Warfare Military engagements
Naval warfare Convoy system Air warfare Cryptography Room 40 Horse use Poison gas Railways Strategic bombing Technology Trench warfare Total war Christmas truce
Last surviving veterans Civilian impact Atrocities Prisoners Casualties Economic history 1918 flu pandemic Destruction of Kalisz Rape of Belgium German
occupation of Belgium German occupation of Luxembourg German occupation of northeastern France Ober Ost Ottoman people Armenian Genocide Assyrian genocide
Pontic Greek genocide Urkun ( Kyrgyzstan ) Blockade of Germany Women Australia Popular culture German prisoners of war in the United States Agreements Partition
of the Ottoman Empire Sykes -- Picot Agreement Agreement of Saint - Jean - de-Maurienne French - Armenian Agreement Damascus Protocol Paris Peace Conference
Venizelos -- Tittoni agreement Treaties Treaty of Brest - Litovsk Treaty of Lausanne Treaty of London Treaty of Neuilly Treaty of St.
</item 9>

<item 10>
Graph query row:
entity: Central Powers
relationship: RELATED_TO
related: Russia
detail: None
</item 10>
=== END ITEM id=5bdfa46a3df716c9 ===

=== ITEM id=c3d4715a2c0824e4 ===
Gold sentences and facts:
Sentence 0: Nigeria is the largest oil and gas producer in Africa . Crude oil from the delta basin comes in two types : light , and comparatively heavy -- the lighter around 36 gravity and the heavier , 20 -- 25 gravity . Both types are paraffinic and low in sulfur .
  Fact 0.0: Nigeria is the largest oil and gas producer in Africa.
  Fact 0.1: Crude oil from the delta basin comes in two types: light, and comparatively heavy.
  Fact 0.2: The lighter type has a gravity of around 36.
  Fact 0.3: The heavier type has a gravity of 20 to 25.
  Fact 0.4: Both types of crude oil from the delta basin are paraffinic.
  Fact 0.5: Both types of crude oil from the delta basin are low in sulfur.

Retrieved items:
<item 1>
Sub-Saharan Africa
Barbary Coast Bashmur Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba
Mountains Nubia The Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf
of Guinea Niger Basin Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern
Highlands Rhodesia North South Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay
Hydra Bay Macro-regions Aethiopia Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater
Middle East Islands of Africa List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland
Northeast Africa Portuguese - speaking African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa
Regions of Asia Central Greater Middle East Aral Sea Aralkum Desert Caspian Sea Dead Sea Sea of Galilee Transoxiana Turan Greater Khorasan
</item 1>

<item 2>
OPEC
Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa
1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 --
1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa
1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,
</item 2>

<item 3>
Sub-Saharan Africa
Australia 31 Af : Sudan 466100 2007 est . Ecuador 33 Af : Eq. Guinea 368500 2007 est . Vietnam 38 Af : DR Congo 261000 2008 est . 39 Af : Gabon 243900 2007 est
. 40 Af : Sth Africa 199100 2007 est . 45 Af : Chad 156000 2008 est . Germany 53 Af : Cameroon 87400 2008 est . France 56 E : France 71400 2007 60 Af : Ivory
Coast 54400 2008 est . _ Af : Africa 10780400 2011 Russia Source : CIA.gov , World Facts Book > Oil exporters . As of 2009 , fifty percent of Africa is rural
with no access to electricity . Africa generates 47 GW of electricity , less than 0.6 % of the global market share . Many countries are affected by power
shortages . Because of rising prices in commodities such as coal and oil , thermal sources of energy are proving to be too expensive for power generation .
</item 3>

<item 4>
Sub-Saharan Africa
Transportation cost is low and no pipelines have to be laid as in Central Asia . Almost all reserves are offshore , so political turmoil within the host country
will not directly interfere with operations . Sub-Saharan oil is viscous , with a very low sulfur content . This quickens the refining process and effectively
reduces costs . New sources of oil are being located in Sub-Saharan Africa more frequently than anywhere else . Of all new sources of oil , 1⁄3 are in
Sub-Saharan Africa . Agriculture ( edit ) Agricultural fields in Rwanda 's Eastern Province Sub-Saharan Africa has more variety of grains than anywhere in the
world . Between 13,000 and 11,000 BCE wild grains began to be collected as a source of food in the cataract region of the Nile , south of Egypt . The collecting
of wild grains as source of food spread to Syria , parts of Turkey , and Iran by the eleventh millennium BCE . By the tenth and ninth millennia southwest Asians
domesticated their wild grains , wheat , and barley after the notion of collecting wild grains spread from the Nile . Numerous crops have been domesticated in
the region and spread to other parts of the world .
</item 4>

<item 5>
Sahara
Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba Mountains Nubia The
Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf of Guinea Niger Basin
Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern Highlands Rhodesia North South
Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay Hydra Bay Macro-regions Aethiopia
Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater Middle East Islands of Africa
List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland Northeast Africa Portuguese - speaking
African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa Coordinates : 23 ° N 13 ° E ﻿ / ﻿ 23 ° N
13 ° E ﻿ / 23 ;
</item 5>

<item 6>
Petroleum industry in Nigeria
An open - air market for illegal crude oil operates off the Niger Delta , called the Togo Triangle . Natural gas ( edit ) Natural gas reserves are well over
5,300 km ( 187 × 10 ^ cu ft ) , the gas reserves are three times as substantial as the crude oil reserves . The biggest natural gas initiative is the Nigerian
Liquified Natural Gas Company , which is operated jointly by several companies and the state . It began exploration and production in 1999 . Chevron is also
attempting to create the Escravos Gas Utilization project which will be capable of producing 4,500,000 m ( 160 × 10 ^ cu ft ) per day . gas reserves . In 2008 ,
the government prepared a Gas Master Plan that was intended to promote natural gas production and encourage the supply of natural gas to domestic power stations
so as to help alleviate the country 's electricity shortages . There is also an export gas pipeline , known as the West African Gas Pipeline , in the works but
has encountered numerous setbacks . The pipeline would allow for transportation of natural gas to Benin , Ghana , Togo , and Cote d'Ivoire .
</item 6>

<item 7>
Petroleum industry in Nigeria
Mobil operates primarily out of the Qua Iboe Terminal in Akwa Ibom State , while Chevron owns the Escravos Terminal located in Delta State and has a storage
capacity of 3.6 million barrels ( 570,000 m ) . Agip operates the Brass Terminal in Brass , a town 113 kilometres ( 70 miles ) southwest of Port Harcourt and
has a storage capacity of 3,558,000 barrels ( 565,700 m ) . Texaco operates the Pennington Terminal . Offshore ( edit ) Oil companies in Africa investigate
offshore production as an alternative area of production . Deepwater production mainly involves underwater drilling that exists 400 metres ( 1,300 ft ) or more
below the surface of the water . By expanding to deep water drilling the possible sources for finding new oil reserves is expanded . Through the introduction of
deep water drilling 50 % more oil is extracted than before the new forms of retrieving the oil . Angola and Nigeria are the largest oil producers in Africa . In
Nigeria , the deepwater sector still has a large avenue to expand and develop .
</item 7>

<item 8>
Petroleum industry in Nigeria
The most productive region of the nation is the coastal Niger Delta Basin in the Niger Delta or `` South - south '' region which encompasses 78 of the 159 oil
fields . Most of Nigeria 's oil fields are small and scattered , and as of 1990 , these small unproductive fields accounted for 62.1 % of all Nigerian
production . This contrasts with the sixteen largest fields which produced 37.9 % of Nigeria 's petroleum at that time . As a result of the numerous small
fields , an extensive and well - developed pipeline network has been engineered to transport the crude oil . Also because of the lack of highly productive
fields , money from the jointly operated ( with the federal government ) companies is constantly directed towards petroleum exploration and production . Nigeria
's petroleum is classified mostly as `` light '' and `` sweet '' , as the oil is largely free of sulphur . Nigeria is the largest producer of sweet oil in OPEC
. This sweet oil is similar in composition to petroleum extracted from the North Sea . This crude oil is known as `` Bonny light '' .
</item 8>

<item 9>
Sub-Saharan Africa
According to the World Bank , the economic growth rate in the region had risen to 4.7 % in 2013 , with a rate of 5.2 % forecasted for 2014 . This continued rise
was attributed to increasing investment in infrastructure and resources as well as steady expenditure per household . Energy and power ( edit ) Main article :
Mineral industry of Africa Oil production by country ( with other key actors of African or oil economy ) Rank Area bb / day Year Like ... _ W : World 85540000
2007 est . 01 E : Russia 9980000 2007 est . 02 Ar : Saudi Arb 9200000 2008 est . 04 As : Libya 4725000 2008 est . Iran 10 Af : Nigeria / Africa 2352000 2011 est
. Norway 15 Af : Algeria 2173000 2007 est . 16 Af : Angola 1910000 2008 est . 17 Af : Egypt 1845000 2007 est . 27 Af : Tunisia 664000 2007 est . Australia 31 Af
: Sudan 466100 2007 est .
</item 9>

<item 10>
Sub-Saharan Africa
Major suppliers are South Africa , Ghana , Zimbabwe , Tanzania , Guinea , and Mali . South Africa had been first in the world in terms of gold production since
1905 , but in 2007 it moved to second place , according to GFMS , the precious metals consultancy . Uranium is major commodity from the region . Significant
suppliers are Niger , Namibia , and South Africa . Namibia was the number one supplier from Sub-Saharan Africa in 2008 . The region produces 49 % of the world
's diamonds . By 2015 , it is estimated that 25 % of North American oil will be from Sub-Saharan Africa , ahead of the Middle East . Sub-Saharan Africa has been
the focus of an intense race for oil by the West , China , India , and other emerging economies , even though it holds only 10 % of proven oil reserves , less
than the Middle East . This race has been referred to as the second Scramble for Africa . All reasons for this global scramble come from the reserves ' economic
benefits . Transportation cost is low and no pipelines have to be laid as in Central Asia .
</item 10>
=== END ITEM id=c3d4715a2c0824e4 ===

=== ITEM id=e7a2d79ee8bb615e ===
Gold sentences and facts:
Sentence 0: Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium NRG Stadium Location Houston , Texas Operated 2006 -- present Conference tie - ins Big 12 , SEC Previous conference tie - ins Big East / C - USA / MWC ( alternating years ) ( 2006 -- 2009 ) Big Ten ( 2010 -- 2013 ) Payout US $ 3 million per team ( as of 2015 ) Preceded by Houston Bowl Sponsors Meineke Car Care ( 2011 -- 2012 ) AdvoCare ( 2014 -- 2016 ) Academy Sports + Outdoors ( 2017 - present ) Former names Texas Bowl ( 2006 -- 2010 ) Meineke Car Care Bowl of Texas ( 2011 -- 2012 ) Texas Bowl ( 2013 ) AdvoCare V100 Texas Bowl ( 2014 -- 2016 ) 2016 matchup Texas A&M vs. Kansas State ( Kansas State 33 -- 28 ) 2017 matchup Missouri vs Texas ( Texas 33 -- 16 )
  Fact 0.0: The Texas Bowl is also called Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium.
  Fact 0.1: The Texas Bowl is held at NRG Stadium.
  Fact 0.2: The Texas Bowl is located in Houston, Texas.
  Fact 0.3: The Texas Bowl has operated from 2006 to the present.
  Fact 0.4: The Texas Bowl's conference tie-ins are the Big 12 and the SEC.
  Fact 0.5: The Texas Bowl's previous conference tie-ins were the Big East, C-USA and MWC (alternating years) from 2006 to 2009.
  Fact 0.6: The Texas Bowl's previous conference tie-in was the Big Ten from 2010 to 2013.
  Fact 0.7: The Texas Bowl's payout is US $3 million per team as of 2015.
  Fact 0.8: The Texas Bowl was preceded by the Houston Bowl.
  Fact 0.9: Meineke Car Care sponsored the Texas Bowl from 2011 to 2012.
  Fact 0.10: AdvoCare sponsored the Texas Bowl from 2014 to 2016.
  Fact 0.11: Academy Sports + Outdoors has sponsored the Texas Bowl since 2017.
  Fact 0.12: A former name of the Texas Bowl was 'Texas Bowl' from 2006 to 2010.
  Fact 0.13: A former name of the Texas Bowl was 'Meineke Car Care Bowl of Texas' from 2011 to 2012.
  Fact 0.14: A former name of the Texas Bowl was 'Texas Bowl' in 2013.
  Fact 0.15: A former name of the Texas Bowl was 'AdvoCare V100 Texas Bowl' from 2014 to 2016.
  Fact 0.16: The 2016 Texas Bowl matchup was Texas A&M vs. Kansas State, which Kansas State won 33-28.
  Fact 0.17: The 2017 Texas Bowl matchup was Missouri vs Texas, which Texas won 33-16.

Retrieved items:
<item 1>
Texas Bowl
In 2007 , TCU took the place of the Big 12 team when Kansas and Oklahoma were put into the BCS , and Houston , a `` home team , '' represented C - USA . The
conferences would receive $612,500 each as per the rules of the agreements as usually , the Big East ( or Big 12 ) would have received $750,000 for playing and
C - USA would have received a $500,000 stipend for their team playing . Issues ( edit ) According to Sports Illustrated , in 2008 the bowl required Western
Michigan University to purchase 11,000 tickets at full price in order to accept the invitation to play in the bowl . The university was only able to sell 548
tickets at that price , forcing it to accept a $462,535 loss , before travel expenses , to pay for the privilege of playing in the bowl . Game results ( edit )
Season Date Time ( CST ) Teams Attendance TV 2006 December 28 7 : 00 PM Rutgers 37 Kansas State 10 52,210 NFL Network 2007 December 28 7 : 00 PM TCU 20 Houston
13 62,
</item 1>

<item 2>
Texas Bowl
wikipedia.org/w/index.php?title=Texas_Bowl&oldid=817734724 '' Categories : College football bowls Texas Bowl American football in Texas Annual events in Texas
Annual sporting events in the United States Sports competitions in Texas Sports in Houston Recurring sporting events established in 2006 Talk Contents About
Wikipedia Français Edit links This page was last edited on 30 December 2017 , at 07 : 17 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
Texas Bowl
It was also announced on December 30 , 2009 , that ESPN Plus would take over as part owner and operator of the game , while Lone Star Sports and Entertainment
will maintain a stake in the bowl , and would be carried on ESPN . Sponsors ( edit ) On April 12 , 2011 , ESPN announced Meineke Car Care signed a three - year
title sponsorship deal beginning in 2011 , changing name of the bowl to the Meineke Car Care Bowl of Texas . On February 12 , 2014 , it was announced that
AdvoCare will be the title sponsor for the bowl game . That sponsorship concluded after the 2016 game . On November 15 , 2017 , Academy Sports + Outdoors became
the new title sponsor of the bowl . Conference tie - ins ( edit ) On May 17 , 2007 , it was announced Conference USA would have a team in the 2007 Texas Bowl .
The Texas Bowl has a rotating commitment with the Big East Conference and Conference USA for 2006 -- 09 while the Big 12 Conference will have a team in all four
of those games .
</item 3>

<item 4>
Texas Bowl
Texas Bowl - wikipedia Texas Bowl Jump to : navigation , search `` AdvoCare V100 Bowl '' and `` AdvoCare Bowl '' redirect here . For the bowl game that AdvoCare
previously sponsored ( 2009 -- 13 ) , see Independence Bowl . Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium NRG Stadium Location Houston , Texas
Operated 2006 -- present Conference tie - ins Big 12 , SEC Previous conference tie - ins Big East / C - USA / MWC ( alternating years ) ( 2006 -- 2009 ) Big Ten
( 2010 -- 2013 ) Payout US $ 3 million per team ( as of 2015 ) Preceded by Houston Bowl Sponsors Meineke Car Care ( 2011 -- 2012 ) AdvoCare ( 2014 -- 2016 )
Academy Sports + Outdoors ( 2017 - present ) Former names Texas Bowl ( 2006 -- 2010 ) Meineke Car Care Bowl of Texas ( 2011 -- 2012 ) Texas Bowl ( 2013 )
AdvoCare V100 Texas Bowl ( 2014 -- 2016 ) 2016 matchup Texas A&M vs.
</item 4>

<item 5>
Texas Bowl
November 15 , 2017 . Jump up ^ Murphy , Austin ; Wetzel , Dan ( 15 November 2010 ) . `` Does It Matter ? '' . Sports Illustrated . p. 47 . External links ( edit
) Official Website Texas Bowl 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 NRG Stadium ( 2006 -- present ) Broadcasters Previous Bowls Bluebonnet Bowl
Houston Bowl NCAA Division I FBS bowl games College Football Playoff Championship Game Cotton Fiesta Orange Peach Rose Sugar Other bowl games Alamo Arizona
Armed Forces Bahamas Belk Birmingham Boca Raton Cactus Camellia Camping World Citrus Cure Dollar General Famous Idaho Potato Foster Farms Frisco Gasparilla
Hawaii Heart of Dallas Holiday Independence Las Vegas Liberty Military Music City New Mexico New Orleans Outback Pinstripe Quick Lane Sun TaxSlayer Texas All -
Star games East -- West Shrine Game NFLPA Collegiate Bowl Senior Bowl Retrieved from `` https://en.wikipedia.org/w/index.
</item 5>

<item 6>
Texas Bowl
097 2008 December 30 7 : 00 PM Rice 38 Western Michigan 14 58,880 2009 December 31 2 : 30 PM Navy 35 Missouri 13 69,441 ESPN December 29 5 : 00 PM Illinois 38
Baylor 14 68,211 2011 December 31 11 : 00 AM Texas A&M 33 Northwestern 22 68,395 2012 December 28 8 : 00 PM Texas Tech 34 Minnesota 31 50,386 2013 December 27 5
: 00 PM Syracuse 21 Minnesota 17 32,327 2014 December 29 8 : 00 PM Arkansas 31 Texas 7 71,115 2015 December 29 8 : 00 PM LSU 56 Texas Tech 27 71,307 2016
December 28 8 : 00 PM Kansas State 33 Texas A&M 28 68,412 2017 December 27 8 : 00 PM Texas 33 Missouri 16 67,
</item 6>

<item 7>
Texas Bowl
In July 2006 , the NFL Network acquired TV rights and naming rights to the bowl . Texas Bowl introduction ( edit ) The Texas Bowl name and logo were officially
unveiled on August 10 , 2006 , at a press conference along with conference affiliations for the bowl spots . The Big 12 , Big East and Conference USA will be
affiliated with the game , as well as Texas Christian University of the Mountain West . The 2006 matchup featured teams from the Big 12 and Big East Conferences
. On December 3 , 2006 , Rutgers accepted an invitation to play Kansas State in the inaugural Texas Bowl . `` We 're ecstatic about having Rutgers , '' Texas
Bowl director David Brady said . `` This is a top - 15 team that was three yards away from a BCS game . We could n't be happier to have them here . '' 2010
marked the eleventh consecutive year a bowl game has played in Houston , and the 40th year overall with a bowl game there ( the Bluebonnet Bowl lasted 29 years
) .
</item 7>

<item 8>
Texas Bowl
Rank Conference Appearances Wins Losses Pct . Big 12 9 5 . 444 T2 SEC . 500 T2 Big Ten . 250 C - USA . 500 T5 ACC 0 1.000 T5 Big East 0 1.000 T5 Independents 0
1.000 T5 Mountain West 0 1.000 T5 MAC 0 . 000 Jump up ^ Rutgers ( 2006 ) appeared as a member of the Big East . Following the 2013 split of the original Big
East along football lines , the FBS schools reorganized as the new American Athletic Conference , which retains the charter of the original Big East . Jump up ^
Navy ( 2009 ) See also ( edit ) List of college bowl games List of Texas Bowl broadcasters References ( edit ) Jump up ^ `` College Bowl Game Payouts '' .
statisticbrain.com. 2015 . Jump up ^ Duarte , Joseph ( 18 April 2006 ) . `` Houston Bowl in jeopardy '' . Houston Chronicle . Retrieved 29 December 2015 . Jump
up ^ `` NFL Network gets bowl game in Houston '' .
</item 8>

<item 9>
Texas Bowl
Since 2017 , the game has been sponsored by Academy Sports + Outdoors . Contents ( hide ) 1 History 1.1 Replacing the Houston Bowl 1.2 Texas Bowl introduction
1.3 Sponsors 1.4 Conference tie - ins 1.5 Issues 2 Game results 3 MVPs 4 Most appearances 5 Appearances by conference 6 See also 7 References 8 External links
History ( edit ) Replacing the Houston Bowl ( edit ) Speculation surfaced questioning the long - term survival of the former Houston Bowl . The three - year
contract with EV1.net expired on December 31 , 2005 , leaving the bowl game without a title sponsor . A college football official told the Houston Chronicle
that the bowl was in danger of ceasing operations , as a result of the game losing its title sponsor and because the Houston Bowl still owed roughly $600,000 to
the Big 12 and Mountain West conferences following the 2005 game . However , the NCAA approved Lone Star Sports & Entertainment , a division of the Houston
Texans , who also play in Reliant Stadium , to take over game management .
</item 9>

<item 10>
Texas Bowl
Kansas State ( Kansas State 33 -- 28 ) 2017 matchup Missouri vs Texas ( Texas 33 -- 16 ) The Texas Bowl , officially the Academy Sports + Outdoors Texas Bowl
for sponsorship purposes , is a post-season NCAA - sanctioned Division I FBS college football bowl game first held in 2006 in Houston , Texas . The bowl
replaced the defunct Houston Bowl , which played annually from 2000 to 2005 , and before that the Bluebonnet Bowl , the first bowl game in Houston , played from
1959 through 1987 . From 2011 to 2012 , the game was sponsored by Meineke Car Care and officially known as the Meineke Car Care Bowl of Texas . The game was
then officially the AdvoCare V100 Texas Bowl from 2014 to 2016 , with AdvoCare as the title sponsor . Since 2017 , the game has been sponsored by Academy Sports
+ Outdoors .
</item 10>
=== END ITEM id=e7a2d79ee8bb615e ===

=== ITEM id=84e10ad7b45e454f ===
Gold sentences and facts:
Sentence 0: Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double Down
  Fact 0.0: The title is Diary of a Wimpy Kid: The Getaway.
  Fact 0.1: The author is Jeff Kinney.
  Fact 0.2: The country is United States.
  Fact 0.3: The language is English.
  Fact 0.4: The series is Diary of a Wimpy Kid.
  Fact 0.5: The genres are children's novel and graphic novel.
  Fact 0.6: The publisher is Amulet Books (US).
  Fact 0.7: The publisher is Puffin Books (UK).
  Fact 0.8: The publication date is November 7, 2017 (worldwide).
  Fact 0.9: The media type is print (hardcover).
  Fact 0.10: The book has 217 story pages, 224 altogether.
  Fact 0.11: The ISBN is 978-1-4197-2545-6.
  Fact 0.12: It is preceded by Double Down.

Retrieved items:
<item 1>
Diary of a Wimpy Kid: The Getaway
Waterstones.com . Retrieved 10 December 2017 . ( hide ) The Diary of a Wimpy Kid series by Jeff Kinney Novels Diary of a Wimpy Kid Rodrick Rules The Last Straw
Dog Days The Ugly Truth Cabin Fever The Third Wheel Hard Luck The Long Haul Old School Double Down The Getaway Characters Greg Heffley Film series Diary of a
Wimpy Kid Rodrick Rules Dog Days The Long Haul Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
Diary of a Wimpy Kid: The Getaway
Please discuss this issue on the article 's talk page . ( December 2017 ) This article needs more links to other articles to help integrate it into the
encyclopedia . Please help improve this article by adding links that are relevant to the context within the existing text . ( December 2017 ) ( Learn how and
when to remove this template message ) ( Learn how and when to remove this template message ) Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country
United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication
date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double
Down Diary of a Wimpy Kid : The Getaway is the twelfth book in the Diary of a Wimpy Kid series by Jeff Kinney .
</item 2>

<item 3>
Diary of a Wimpy Kid: The Getaway
Diary of a Wimpy Kid : the Getaway - wikipedia Diary of a Wimpy Kid : the Getaway Jump to : navigation , search ( hide ) This article has multiple issues .
Please help improve it or discuss these issues on the talk page . ( Learn how and when to remove these template messages ) This article consists almost entirely
of a plot summary . It should be expanded to provide more balanced coverage that includes real - world context . Please edit the article to focus on discussing
the work rather than merely reiterating the plot . ( August 2016 ) ( Learn how and when to remove this template message ) This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
December 2017 ) ( Learn how and when to remove this template message ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page .
</item 3>

<item 4>
Diary of a Wimpy Kid (book series)
2016 ) Diary of a Wimpy Kid : The Getaway ( November 7 , 2017 ) Characters ( edit ) Main article : List of Diary of a Wimpy Kid characters Greg Heffley is a
wimpy student , who is the main character and narrator of the series . Rowley Jefferson is Greg 's childish best friend . Rodrick Heffley is Greg and Manny 's
older brother , who is rude , and the drummer in his self - titled band , Löded Diper . He usually is noted for playing tricks on Greg . Manny Heffley is Greg
and Rodrick 's spoiled younger brother , who is known to be a tattletale . Susan Heffley is Greg 's mother . She often embarrasses Greg . Frank Heffley is Greg
's father . He always wants Greg to do sports and `` to become a man , '' and does n't like giving up . Gramma is Greg 's grandmother on his mother 's side .
Fregley is Greg 's weird neighbor and classmate .
</item 4>

<item 5>
Diary of a Wimpy Kid: The Getaway
Release date ( edit ) The Getaway is the 12th book in the series , and was released everywhere except Canada and China on November 7 , 2017 . Reception ( edit )
A week since its release , it has received 4.5 / 5 stars on Waterstones with 786 reviews . References ( edit ) Jump up ^ Wimpy Kid ( 27 April 2017 ) . `` Wimpy
Kid LIVE Special 10th Anniversary Presentation and Cover Reveal of Book 12 ! '' . YouTube . Retrieved 10 December 2017 . Jump up ^ `` Cover Of New Diary Of A
Wimpy Kid Book Revealed By Jeff Kinney To Fans Worldwide At Live streamed Carnegie Hall Event Hosted By ABRAMS '' . Tristateupdate.com . Retrieved 10 December
2017 . Jump up ^ `` Diary of a Wimpy Kid : The Getaway by Jeff Kinney -- Waterstones '' . Waterstones.com . Retrieved 10 December 2017 .
</item 5>

<item 6>
Diary of a Wimpy Kid: The Getaway
The book was unveiled during the 2017 Diary of a Wimpy Kid Virtually Live Event which was live streamed via YouTube as part of the 10th anniversary of the first
book . The book was published and released on November 7 , 2017 . Contents ( hide ) 1 Plot 2 Release date 3 Reception 4 References Plot ( edit ) After many
frustrations , Greg Heffley 's parents decide to skip Christmas and travel by plane to Isla De Corales , the hotel they stayed at while on their honeymoon much
to Greg 's dismay . On Christmas Eve they drive to the airport , but are delayed by a series of mishaps . They encounter holiday traffic but arrive at the
airport . As they fail to unload in time at the drop off zone , Frank and Greg have to return later with the luggage . Greg steps in a slush puddle and they end
up missing the shuttle . In the airport , the family 's suitcases are too heavy , so they stuff some clothes into their carry on baggage .
</item 6>

<item 7>
Diary of a Wimpy Kid: The Getaway
The couple sit on either side of him with their baby . Greg becomes paranoid when he watches the plane 's safety video and begins to ask questions about
unfortunate events . Greg is very aggravated of the other passengers because most are n't watching the safety video . Mid flight , he feels boxed in when the
people in front recline and the person behind slides his feet on Greg 's armrest . Greg tries to recline his chair too but when he asks the flight attendant how
, he learns that he ca n't because the row behind him is an emergency exit . He tries to swap seats with Frank but Rodrick beats him to it . He stays in the
bathroom because the couple 's baby started to get annoying , but then he returns to his seat after experiencing turbulence . He somehow was able to sleep for
the rest of the flight . As they land on Christmas Day , Greg begins to understand why the family came there . They touch down and began to head to the resort ,
Isla de Corales . He is even excited by the activities the resort offer which he sees on the bus .
</item 7>

<item 8>
Diary of a Wimpy Kid: The Getaway
Rodrick then goes back to the teen zone . Greg takes Manny to the Toddler 's Cabana , but Manny gets out of going there because he is not potty trained . He
puts Manny in the pool , but has to get him out again later . Susan booked a charter cruise for the whole family due in thirty minutes . On the cruise ( which
turns out to be a regular boat ) , the captain offers them to go scuba diving . Greg is freaked out by all of the deadly animals , including the venomous box
jellyfish , and only stays in for one photo . As he surfaces , he swallows a big gulp of water . He thinks a seahorse was in there too : so they return to the
resort to see a doctor . Frank and Susan were both seasick , so they just return to the poolside . They see the Director of Fun , who keeps circling their chair
, but suddenly stops at a plastic bag in Manny 's bucket . It is actually a box jellyfish which Manny smuggled from the scuba diving , in which the Director of
Fun launches it back into the pool in fear .
</item 8>

<item 9>
Diary of a Wimpy Kid: The Getaway
He is even excited by the activities the resort offer which he sees on the bus . However , Frank and Susan are disappointed that the resort has since changed ,
and it is now split with a family side ( the Mild Side ) and a couple side ( the Wild Side ) . The family also discovers that they got the wrong luggage that is
locked , and their luggage was sent back home . They only have the clothes they put in their carry on luggage . They attempt to buy some new clothes , but Frank
wo n't buy them as they are too expensive . The family goes to explore the resort and end up at the pool . The Director of Fun shows up , however , and gets
everybody moving . Greg is forced to take part in two games . After this , Greg sees an iguana by the pool and is worried , so the family goes to get some lunch
. Greg and his family try to eat outside ; however , too many birds and bugs keep ruining their meals . Frank contracts a stomach bug from his water , so they
head back to the suite . The next day , Greg uses the shower which is outside .
</item 9>

<item 10>
Diary of a Wimpy Kid: The Getaway
They enter the line for security , but a child unclips a barrier separating the lines , and everybody gets scrambled . They go through security and are delayed
again by somebody who has to remove all of his piercings . Manny then gets his toy stuck in an escalator , delaying them more . As a result , the family panic ,
grab some luggage and dash to the gate . It turns out their flight is delayed . Greg goes to dry his sock but destroys it while doing so . He hears that the
flight is overbooked , and tries to stop his family boarding in exchange of a hotel stay and $300 ( soon $500 ) , but is stopped by Susan . A replacement crew
for the plane is required and they finally board . Greg is first impressed by the plane 's first class seats before he discovers that he has to sit in economy .
However , Frank collected enough air miles because of his business trips to sit in first class . In economy , Mom , Rodrick , and Manny are seated in the right
row . Greg finds out he has three seats to himself , until a couple with a baby arrives late on board .
</item 10>
=== END ITEM id=84e10ad7b45e454f ===

=== ITEM id=c4c46d009857bc52 ===
Gold sentences and facts:
Sentence 0: When following the western side of the African coastline from the equator , however , the Cape of Good Hope marks the point where a ship begins to travel more eastward than southward . Thus , the first modern rounding of the cape in 1488 by Portuguese explorer Bartolomeu Dias was a milestone in the attempts by the Portuguese to establish direct trade relations with the Far East ( although Herodotus mentioned a claim that the Phoenicians had done so far earlier ) . Dias called the cape Cabo das Tormentas ( `` Cape of Storms '' ; Dutch : Stormkaap ) , which was the original name of the `` Cape of Good Hope '' .
  Fact 0.0: When following the western side of the African coastline from the equator, the Cape of Good Hope marks the point where a ship begins to travel more eastward than southward.
  Fact 0.1: Portuguese explorer Bartolomeu Dias made the first modern rounding of the Cape of Good Hope in 1488.
  Fact 0.2: The 1488 rounding of the cape by Bartolomeu Dias was a milestone in the attempts by the Portuguese to establish direct trade relations with the Far East.
  Fact 0.3: Herodotus mentioned a claim that the Phoenicians had rounded the cape far earlier.
  Fact 0.4: Bartolomeu Dias called the cape Cabo das Tormentas.
  Fact 0.5: Cabo das Tormentas means 'Cape of Storms' in English.
  Fact 0.6: The Dutch name for Cabo das Tormentas is Stormkaap.
  Fact 0.7: Cabo das Tormentas was the original name of the Cape of Good Hope.

Retrieved items:
<item 1>
Sub-Saharan Africa
Monomotapa was a medieval kingdom ( c. 1250 -- 1629 ) , which existed between the Zambezi and Limpopo rivers of Southern Africa in the territory of modern - day
Zimbabwe and Mozambique . Its old capital was located at Great Zimbabwe . In 1487 , Bartolomeu Dias became the first European to reach the southernmost tip of
Africa . In 1652 , a victualling station was established at the Cape of Good Hope by Jan van Riebeeck on behalf of the Dutch East India Company . For most of
the 17th and 18th centuries , the slowly expanding settlement was a Dutch possession . Great Britain seized the Cape of Good Hope area in 1795 , ostensibly to
prevent it from falling into the hands of the French but also to use Cape Town in particular as a stop on the route to Australia and India . It was later
returned to the Dutch in 1803 , but soon afterwards the Dutch East India Company declared bankruptcy , and the British annexed the Cape Colony in 1806 . The
Zulu Kingdom was a Southern African tribal state in what is now KwaZulu - Natal in southeastern South Africa .
</item 1>

<item 2>
Cape of Good Hope
Somewhere along the coast of East Africa , he found the remains of the ship . Due to its appearance and the story told by the natives , Eudoxus concluded that
the ship was from Gades and had sailed anti-clockwise around Africa , passing the Cape and entering the Indian Ocean . This inspired him to repeat the voyage
and attempt a circumnavigation of the continent . Organising the expedition on his own account he set sail from Gades and began to work down the African coast .
The difficulties were too great , however , and he was obliged to return to Europe . After this failure he again set out to circumnavigate Africa . His eventual
fate is unknown . Although some , such as Pliny , claimed that Eudoxus did achieve his goal , the most probable conclusion is that he perished on the journey .
In the 1450 Fra Mauro map , the Indian Ocean is depicted as connected to the Atlantic .
</item 2>

<item 3>
Cape of Good Hope
Legends ( edit ) The Cape of Good Hope is the legendary home of The Flying Dutchman . Crewed by tormented and damned ghostly sailors , it is doomed forever to
beat its way through the adjacent waters without ever succeeding in rounding the headland . Adamastor is a Greek - type mythological character invented by the
Portuguese poet Luís de Camões in his epic poem Os Lusíadas ( first printed in 1572 ) , as a symbol of the forces of nature Portuguese navigators had to
overcome during their discoveries and more specifically of the dangers Portuguese sailors faced when trying to round the Cape of Storms . See also ( edit ) Cape
Peninsula Cape Horn , its South American counterpart Cape Point Western Cape History of Cape Colony Simon 's Town Postage stamps and postal history of the Cape
of Good Hope References ( edit ) Jump up ^ Every word in isolation ( ˌkaːp də ˌɣudə ˈɦoːp ) . Jump up ^ `` Cape of Good Hope , South Africa - 360 ° Aerial
Panoramas '' . Jump up ^ The first circumnavigation of Africa .
</item 3>

<item 4>
Cape of Good Hope
Fra Mauro puts the following inscription by the southern tip of Africa , which he names the `` Cape of Diab '' , describing the exploration by a ship from the
East around 1420 : Detail of the Fra Mauro Map describing the construction of the junks that navigate in the Indian Ocean . `` Around 1420 a ship , or junk ,
from India crossed the Sea of India towards the Island of Men and the Island of Women , off Cape Diab , between the Green Islands and the shadows . It sailed
for 40 days in a south - westerly direction without ever finding anything other than wind and water . According to these people themselves , the ship went some
2,000 miles ahead until - once favourable conditions came to an end - it turned round and sailed back to Cape Diab in 70 days '' . `` The ships called junks (
lit . `` Zonchi '' ) that navigate these seas carry four masts or more , some of which can be raised or lowered , and have 40 to 60 cabins for the merchants and
only one tiller .
</item 4>

<item 5>
Cape of Good Hope
Contemporary scientific knowledge instead states the southernmost point of Africa is Cape Agulhas about 150 kilometres ( 90 mi ) to the east - southeast . The
currents of the two oceans meet at the point where the warm - water Agulhas current meets the cold water Benguela current and turns back on itself . That
oceanic meeting point fluctuates between Cape Agulhas and Cape Point ( about 1.2 kilometres ( 0.75 mi ) east of the Cape of Good Hope ) . When following the
western side of the African coastline from the equator , however , the Cape of Good Hope marks the point where a ship begins to travel more eastward than
southward . Thus , the first modern rounding of the cape in 1488 by Portuguese explorer Bartolomeu Dias was a milestone in the attempts by the Portuguese to
establish direct trade relations with the Far East ( although Herodotus mentioned a claim that the Phoenicians had done so far earlier ) . Dias called the cape
Cabo das Tormentas ( `` Cape of Storms '' ; Dutch : Stormkaap ) , which was the original name of the `` Cape of Good Hope '' .
</item 5>

<item 6>
Cape of Good Hope
Jump up ^ The first circumnavigation of Africa . livius.org Jump up ^ Sarah Mytton Maury ( 1848 ) . Englishwoman In America . p. 33 . Jump up ^ Along the
Clipper Way , Francis Chichester ; page 78 . Hodder & Stoughton , 1966 . ISBN 0 - 340 - 00191 - 7 Jump up ^ Map of the Park , showing the Cape of Good Hope
section ( retrieved 27 March 2010 ) . Jump up ^ Tozer , Henry F. ( 1997 ) . History of Ancient Geography . Biblo & Tannen . pp. 189 -- 190 . ISBN 0 - 8196 -
0138 - 1 . ; online at Google Books Jump up ^ Tozer , Henry F. ( 1997 ) . History of Ancient Geography . Biblo & Tannen . pp. xxiii . ISBN 0 - 8196 - 0138 - 1 .
; online at Google Books Jump up ^ Marco Polo , p.
</item 6>

<item 7>
Cape of Good Hope
They can navigate without a compass , because they have an astrologer , who stands on the side and , with an astrolabe in hand , gives orders to the navigator
`` . ( Text from the Fra Mauro map , 09 - P25 . ) Fra Mauro explained that he obtained the information from `` a trustworthy source '' , who traveled with the
expedition , possibly the Venetian explorer Niccolò da Conti who happened to be in Calicut , India at the time the expedition left : `` What is more , I have
spoken with a person worthy of trust , who says that he sailed in an Indian ship caught in the fury of a tempest for 40 days out in the Sea of India , beyond
the Cape of Soffala and the Green Islands towards west - southwest ; and according to the astrologers who act as their guides , they had advanced almost 2,000
miles . Thus one can believe and confirm what is said by both these and those , and that they had therefore sailed 4,000 miles '' .
</item 7>

<item 8>
Cape of Good Hope
It remained a British colony until being incorporated into the independent Union of South Africa in 1910 ( now known as the Republic of South Africa ) . The
Portuguese government erected two navigational beacons , Dias Cross and Gama Cross , to commemorate Vasco da Gama and Bartolomeu Dias as explorers who as
mentioned were the first explorers to reach the cape . When lined up , the crosses point to Whittle Rock ( 34 ° 21 ′ 24.63 '' S 18 ° 28 ′ 26.36 '' E ﻿ / ﻿
34.3568417 ° S 18.4739889 ° E ﻿ / - 34.3568417 ; 18.4739889 ) , a large , permanently submerged shipping hazard in False Bay . Two other beacons in Simon 's
Town provide the intersection . Geography ( edit ) Sign at the Cape of Good Hope The Cape of Good Hope is at the southern tip of the Cape Peninsula , about 2.3
kilometres ( 1.4 mi ) west and a little south of Cape Point on the south - east corner .
</item 8>

<item 9>
Cape of Good Hope
Fra Mauro also comments that the account of the expedition , together with the relation by Strabo of the travels of Eudoxus of Cyzicus from Arabia to Gibraltar
through the southern Ocean in Antiquity , led him to believe that the Indian Ocean was not a closed sea and that Africa could be circumnavigated by her southern
end ( Text from Fra Mauro map , 11 , G2 ) . This knowledge , together with the map depiction of the African continent , probably encouraged the Portuguese to
intensify their effort to round the tip of Africa . Map showing the Cape Peninsula , illustrating the position of the Cape of Good Hope . The main mountains and
their peaks , including Table Mountain , and its relation to the City of Cape Town are shown . Map of the Cape of Good Hope and Cape Agulhas the southernmost
point of Africa . Main article : Cape Colony Reproduction of the Cross of Vasco da Gama at the Cape of Good Hope .
</item 9>

<item 10>
Cape of Good Hope
In the Early Modern Era , the first European to reach the cape was the Portuguese explorer Bartolomeu Dias on 12 March 1488 , who named it the `` Cape of Storms
'' ( Cabo das Tormentas ) . It was later renamed by John II of Portugal as `` Cape of Good Hope '' ( Cabo da Boa Esperança ) because of the great optimism
engendered by the opening of a sea route to India and the East . The land around the cape was home to the Khoikhoi people when the Dutch first settled there in
1652 . The Khoikhoi had arrived in these parts about fifteen hundred years before . They were called Hottentots by the Dutch , a term that has now come to be
regarded as pejorative . Dutch colonial administrator Jan van Riebeeck established a resupply camp for the Dutch East India Company some 50 km north of the cape
in Table Bay on 6 April 1652 and this eventually developed into Cape Town . Supplies of fresh food were vital on the long journey around Africa and Cape Town
became known as `` The Tavern of the Seas '' .
</item 10>
=== END ITEM id=c4c46d009857bc52 ===

=== ITEM id=3d31a5406fa87e0a ===
Gold sentences and facts:
Sentence 0: Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington , D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game - winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of seven to win the series when trailing three games to one . 1926 St. Louis Cardinals New York Yankees 3 -- 2 Yankee Stadium , the Bronx , New York City Grover Cleveland Alexander strikes out Tony Lazzeri with the bases loaded to end the seventh and preserve the Cardinals lead . Two innings later Babe Ruth is caught stealing to end the World Series . 1931 St. Louis Cardinals Philadelphia Athletics 4 -- 2 St. Louis , Missouri Last postseason game for the Athletics while based in Philadelphia . 1934 St. Louis Cardinals Detroit Tigers 11 -- 0 Navin Field , Detroit Dizzy Dean 's second win , a six - hit shutout on one day of rest , gives the Cardinals the title . 1940 Cincinnati Reds Detroit Tigers 2 -- 1 Crosley Field , Cincinnati This game was over in a snappy 1 : 47 while a sacrifice bunt and sacrifice fly scored the winning runs . 1945 Detroit Tigers Chicago Cubs 9 -- 3 Wrigley Field , Chicago Cubs last World Series appearance until 2016 . See also : Curse of the Billy Goat 1946 St. Louis Cardinals Boston Red Sox 4 -- 3 Sportsman 's Park III , St. Louis , Missouri Enos Slaughter 's famous mad dash scored the deciding run to win the game and the series . 1947 New York Yankees Brooklyn Dodgers 5 -- 2 Yankee Stadium , the Bronx , New York City This was the Series that featured Cookie Lavagetto 's pinch - hit that spoiled Bill Bevens ' no - hit bid and Al Gionfriddo 's catch to rob Joe DiMaggio of a home run . 1952 New York Yankees Brooklyn Dodgers 4 -- 2 Ebbets Field , Brooklyn , New York City Billy Martin 's game saving infield catch preserves the Yankees lead and helps win the game and the series . 1955 Brooklyn Dodgers New York Yankees 2 -- 0 Yankee Stadium , the Bronx , New York City Dodgers ' only championship in Brooklyn . The home team won all games of the series , except for game seven . 1956 New York Yankees Brooklyn Dodgers 9 -- 0 Ebbets Field , Brooklyn , New York City Jackie Robinson 's final Major League game . Last postseason game at Ebbets Field and Dodgers ' last postseason game before moving to Los Angeles . The home team won all games of the series , except for game seven . 1957 Milwaukee Braves New York Yankees 5 -- 0 Yankee Stadium , the Bronx , New York City The Braves ' only championship in Milwaukee . 1958 New York Yankees Milwaukee Braves 6 -- 2 County Stadium , Milwaukee The Yankees win their 18th World Series title after being down three games to one . This was the Braves ' last postseason game in Milwaukee . 1960 Pittsburgh Pirates New York Yankees 10 -- 9 Forbes Field , Pittsburgh Bill Mazeroski hit the walk - off home run to win the Pirates the championship . Last postseason game in Forbes Field . Last postseason game for Yankees ' manager Casey Stengel . Highest scoring game seven . 1962 New York Yankees San Francisco Giants 1 -- 0 Candlestick Park , San Francisco With runners on second and third and two outs , Willie McCovey 's line drive to Bobby Richardson ended the game . First game seven that ended with a 1 - 0 score . 1964 St. Louis Cardinals New York Yankees 7 -- 5 Busch Stadium I , St. Louis , Missouri The Yankees ' last World Series appearance until 1976 . Last postseason game at Busch Stadium I . 1965 Los Angeles Dodgers Minnesota Twins 2 -- 0 Metropolitan Stadium , Bloomington , Minnesota Pitching through immense pain and on two days rest , Sandy Koufax throws a 3 hit shutout to win the Series and World Series MVP . The home team won all games of the series , except for game seven . 1967 St. Louis Cardinals Boston Red Sox 7 -- 2 Fenway Park , Boston Bob Gibson wins his third game of the Series and also homers . 1968 Detroit Tigers St. Louis Cardinals 4 -- 1 Busch Stadium II , St. Louis , Missouri Mickey Lolich out - duels Bob Gibson . The Tigers rallied from a 3 - 1 deficit to win the series . First game seven loss by the Cardinals in the World Series . 1971 Pittsburgh Pirates Baltimore Orioles 2 -- 1 Memorial Stadium , Baltimore The home team won all games of the series , except for game seven . 1972 Oakland Athletics Cincinnati Reds 3 -- 2 Riverfront Stadium , Cincinnati The A 's first championship in Oakland . The A 's were the last American League team to win a World Series game seven on the road until 2017 . 1973 Oakland Athletics New York Mets 5 -- 2 Oakland -- Alameda County Coliseum , Oakland , California Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game to set the tone for victory and the series . Cincinnati Reds Boston Red Sox 4 -- 3 Fenway Park , Boston One night after Carlton Fisk 's famous home run , the Reds rebound to take the Series . 1979 Pittsburgh Pirates Baltimore Orioles 4 -- 1 Memorial Stadium , Baltimore The Pittsburgh Pirates , powered by NLCS and World Series MVP Willie Stargell , rally from a 3 - 1 deficit to win the series . The Pirates were the last team to win a road game seven until 2014 . 1982 St. Louis Cardinals Milwaukee Brewers 6 -- 3 Busch Stadium II , St. Louis , Missouri Joaquín Andújar earns his second win of the Series and Bruce Sutter his second save . 1985 Kansas City Royals St. Louis Cardinals 11 -- 0 Royals Stadium , Kansas City , Missouri Second World Series held entirely in the state of Missouri . The Royals also won the 1985 American League Championship Series in seven games . Royals became first team to win the World Series after losing the first two games at home . The Royals were the last team to come back from 3 -- 1 down to win a title , in any of the five major men 's professional sports leagues , until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3 -- 1 . The 1985 Royals were the last team to win a World Series after trailing 3 -- 1 until the Cubs did so in 2016 . 1986 New York Mets Boston Red Sox 8 -- 5 Shea Stadium , Flushing , Queens , New York Mets become the first team to be one strike away from elimination and come back to win the series . They also become the second team to win the series after losing the first two at home . Minnesota Twins St. Louis Cardinals 4 -- 2 Hubert H. Humphrey Metrodome , Minneapolis The home team won all seven games of the series for the first time in MLB history . 1991 Minnesota Twins Atlanta Braves 1 -- 0 F / 10 Hubert H. Humphrey Metrodome , Minneapolis Like in 1987 , the home team won all games of the series . The Twins won in the bottom of the 10th on a walk - off single by Gene Larkin scoring Dan Gladden after ​ 9 ⁄ scoreless innings of play . Jack Morris goes all ten innings for the Twins and is named the Series MVP . Twins Centerfielder Kirby Puckett hit a famous walk - off home run in the bottom of the eleventh inning of the previous night 's game to force this decisive game . 1997 Florida Marlins Cleveland Indians 3 -- 2 F / 11 Pro Player Stadium , Miami Gardens , Florida Florida becomes the first wild card team to win the World Series . The Marlins rallied for one run in the bottom of the ninth to tie and won in the eleventh on an Édgar Rentería RBI single off Charles Nagy . 2001 Arizona Diamondbacks New York Yankees 3 -- 2 Bank One Ballpark , Phoenix , Arizona Arizona becomes the fastest expansion team to win the World Series , rallying for two runs in the bottom of the ninth for a come - from - behind win , capped off by Luis Gonzalez hitting the winning RBI single off closer Mariano Rivera . The home team won every game of the series . 2002 Anaheim Angels San Francisco Giants 4 -- 1 Edison International Field , Anaheim , California Both World Series teams were wild card entrants into the playoffs . John Lackey became the second rookie pitcher to win a World Series game seven . 2011 St. Louis Cardinals Texas Rangers 6 -- 2 Busch Stadium , St. Louis , Missouri Cardinals became the second team to be one strike away from elimination and come back to win the series , which they did twice . Just like the 1991 series , a walk - off home run in the bottom of the eleventh inning of game six , forced this decisive game with David Freese providing the fire power . Most recent game seven won by the home team . 2014 San Francisco Giants Kansas City Royals 3 -- 2 Kauffman Stadium , Kansas City , Missouri Like in 2002 , both World Series teams were wild card entrants into the playoffs . The Giants became the first NL team to win a game seven on the road since 1979 . NLCS MVP and eventual World Series MVP Madison Bumgarner , normally a starter , pitched five scoreless innings of relief on two days rest to save the Giants ' third title in five years . 2016 Chicago Cubs Cleveland Indians 8 -- 7 F / 10 Progressive Field , Cleveland , Ohio Chicago Cubs ' first World Series win since 1908 . Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City Royals in 1985 .
  Fact 0.0: 1909 World Series: Pittsburgh Pirates beat Detroit Tigers 8-0 at Bennett Park, Detroit.
  Fact 0.1: 1909 World Series: Rookie Babe Adams tossed a six-hit shutout for his third win of the series.
  Fact 0.2: 1912 World Series: Boston Red Sox beat New York Giants 3-2 in 10 innings (F/10) at Fenway Park, Boston.
  Fact 0.3: 1912 World Series: The decisive contest was actually game eight, as game two had ended in a tie.
  Fact 0.4: 1924 World Series: Washington Senators beat New York Giants 4-3 in 12 innings (F/12) at Griffith Stadium, Washington, D.C.
  Fact 0.5: 1924 World Series: It was the longest game seven in innings.
  Fact 0.6: 1924 World Series: In the 12th inning, Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy.
  Fact 0.7: 1924 World Series: Muddy Ruel scored on a game-winning RBI double by Earl McNeely.
  Fact 0.8: 1925 World Series: Pittsburgh Pirates beat Washington Senators 9-7 at Forbes Field, Pittsburgh.
  Fact 0.9: 1925 World Series: Pittsburgh became the first team in a best-of-seven series to win when trailing three games to one.
  Fact 0.10: 1926 World Series: St. Louis Cardinals beat New York Yankees 3-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.11: 1926 World Series: Grover Cleveland Alexander struck out Tony Lazzeri with the bases loaded to end the seventh inning and preserve the Cardinals lead.
  Fact 0.12: 1926 World Series: Two innings later, Babe Ruth was caught stealing to end the World Series.
  Fact 0.13: 1931 World Series: St. Louis Cardinals beat Philadelphia Athletics 4-2 at St. Louis, Missouri.
  Fact 0.14: 1931 World Series: It was the last postseason game for the Athletics while based in Philadelphia.
  Fact 0.15: 1934 World Series: St. Louis Cardinals beat Detroit Tigers 11-0 at Navin Field, Detroit.
  Fact 0.16: 1934 World Series: Dizzy Dean's second win, a six-hit shutout on one day of rest, gave the Cardinals the title.
  Fact 0.17: 1940 World Series: Cincinnati Reds beat Detroit Tigers 2-1 at Crosley Field, Cincinnati.
  Fact 0.18: 1940 World Series: The game was over in a snappy 1:47.
  Fact 0.19: 1940 World Series: A sacrifice bunt and a sacrifice fly scored the winning runs.
  Fact 0.20: 1945 World Series: Detroit Tigers beat Chicago Cubs 9-3 at Wrigley Field, Chicago.
  Fact 0.21: 1945 World Series: It was the Cubs' last World Series appearance until 2016.
  Fact 0.22: 1945 World Series: The row has a see-also note for the Curse of the Billy Goat.
  Fact 0.23: 1946 World Series: St. Louis Cardinals beat Boston Red Sox 4-3 at Sportsman's Park III, St. Louis, Missouri.
  Fact 0.24: 1946 World Series: Enos Slaughter's famous mad dash scored the deciding run to win the game and the series.
  Fact 0.25: 1947 World Series: New York Yankees beat Brooklyn Dodgers 5-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.26: 1947 World Series: This was the Series that featured Cookie Lavagetto's pinch-hit that spoiled Bill Bevens' no-hit bid.
  Fact 0.27: 1947 World Series: Al Gionfriddo's catch robbed Joe DiMaggio of a home run.
  Fact 0.28: 1952 World Series: New York Yankees beat Brooklyn Dodgers 4-2 at Ebbets Field, Brooklyn, New York City.
  Fact 0.29: 1952 World Series: Billy Martin's game-saving infield catch preserved the Yankees lead and helped win the game and the series.
  Fact 0.30: 1955 World Series: Brooklyn Dodgers beat New York Yankees 2-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.31: 1955 World Series: It was the Dodgers' only championship in Brooklyn.
  Fact 0.32: 1955 World Series: The home team won all games of the series except game seven.
  Fact 0.33: 1956 World Series: New York Yankees beat Brooklyn Dodgers 9-0 at Ebbets Field, Brooklyn, New York City.
  Fact 0.34: 1956 World Series: It was Jackie Robinson's final Major League game.
  Fact 0.35: 1956 World Series: It was the last postseason game at Ebbets Field.
  Fact 0.36: 1956 World Series: It was the Dodgers' last postseason game before moving to Los Angeles.
  Fact 0.37: 1956 World Series: The home team won all games of the series except game seven.
  Fact 0.38: 1957 World Series: Milwaukee Braves beat New York Yankees 5-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.39: 1957 World Series: It was the Braves' only championship in Milwaukee.
  Fact 0.40: 1958 World Series: New York Yankees beat Milwaukee Braves 6-2 at County Stadium, Milwaukee.
  Fact 0.41: 1958 World Series: The Yankees won their 18th World Series title after being down three games to one.
  Fact 0.42: 1958 World Series: It was the Braves' last postseason game in Milwaukee.
  Fact 0.43: 1960 World Series: Pittsburgh Pirates beat New York Yankees 10-9 at Forbes Field, Pittsburgh.
  Fact 0.44: 1960 World Series: Bill Mazeroski hit the walk-off home run to win the Pirates the championship.
  Fact 0.45: 1960 World Series: It was the last postseason game in Forbes Field.
  Fact 0.46: 1960 World Series: It was the last postseason game for Yankees' manager Casey Stengel.
  Fact 0.47: 1960 World Series: It was the highest scoring game seven.
  Fact 0.48: 1962 World Series: New York Yankees beat San Francisco Giants 1-0 at Candlestick Park, San Francisco.
  Fact 0.49: 1962 World Series: With runners on second and third and two outs, Willie McCovey's line drive to Bobby Richardson ended the game.
  Fact 0.50: 1962 World Series: It was the first game seven that ended with a 1-0 score.
  Fact 0.51: 1964 World Series: St. Louis Cardinals beat New York Yankees 7-5 at Busch Stadium I, St. Louis, Missouri.
  Fact 0.52: 1964 World Series: It was the Yankees' last World Series appearance until 1976.
  Fact 0.53: 1964 World Series: It was the last postseason game at Busch Stadium I.
  Fact 0.54: 1965 World Series: Los Angeles Dodgers beat Minnesota Twins 2-0 at Metropolitan Stadium, Bloomington, Minnesota.
  Fact 0.55: 1965 World Series: Sandy Koufax pitched through immense pain and on two days rest.
  Fact 0.56: 1965 World Series: Sandy Koufax threw a 3-hit shutout to win the Series.
  Fact 0.57: 1965 World Series: Sandy Koufax was named World Series MVP.
  Fact 0.58: 1965 World Series: The home team won all games of the series except game seven.
  Fact 0.59: 1967 World Series: St. Louis Cardinals beat Boston Red Sox 7-2 at Fenway Park, Boston.
  Fact 0.60: 1967 World Series: Bob Gibson wins his third game of the Series.
  Fact 0.61: 1967 World Series: Bob Gibson also homers in this game.
  Fact 0.62: 1968 World Series: Detroit Tigers beat St. Louis Cardinals 4-1 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.63: 1968 World Series: Mickey Lolich out-dueled Bob Gibson.
  Fact 0.64: 1968 World Series: The Tigers rallied from a 3-1 deficit to win the series.
  Fact 0.65: 1968 World Series: It was the first game seven loss by the Cardinals in the World Series.
  Fact 0.66: 1971 World Series: Pittsburgh Pirates beat Baltimore Orioles 2-1 at Memorial Stadium, Baltimore.
  Fact 0.67: 1971 World Series: The home team won all games of the series except game seven.
  Fact 0.68: 1972 World Series: Oakland Athletics beat Cincinnati Reds 3-2 at Riverfront Stadium, Cincinnati.
  Fact 0.69: 1972 World Series: It was the A's first championship in Oakland.
  Fact 0.70: 1972 World Series: The A's were the last American League team to win a World Series game seven on the road until 2017.
  Fact 0.71: 1973 World Series: Oakland Athletics beat New York Mets 5-2 at Oakland-Alameda County Coliseum, Oakland, California.
  Fact 0.72: 1973 World Series: Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game.
  Fact 0.73: 1973 World Series: Those home runs set the tone for victory and the series.
  Fact 0.74: A row with no year listed: Cincinnati Reds beat Boston Red Sox 4-3 at Fenway Park, Boston.
  Fact 0.75: A row with no year listed: It was one night after Carlton Fisk's famous home run.
  Fact 0.76: A row with no year listed: The Reds rebounded to take the Series.
  Fact 0.77: 1979 World Series: Pittsburgh Pirates beat Baltimore Orioles 4-1 at Memorial Stadium, Baltimore.
  Fact 0.78: 1979 World Series: The Pirates were powered by NLCS and World Series MVP Willie Stargell.
  Fact 0.79: 1979 World Series: The Pirates rallied from a 3-1 deficit to win the series.
  Fact 0.80: 1979 World Series: The Pirates were the last team to win a road game seven until 2014.
  Fact 0.81: 1982 World Series: St. Louis Cardinals beat Milwaukee Brewers 6-3 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.82: 1982 World Series: Joaquín Andújar earned his second win of the Series.
  Fact 0.83: 1982 World Series: Bruce Sutter earned his second save.
  Fact 0.84: 1985 World Series: Kansas City Royals beat St. Louis Cardinals 11-0 at Royals Stadium, Kansas City, Missouri.
  Fact 0.85: 1985 World Series: It was the second World Series held entirely in the state of Missouri.
  Fact 0.86: 1985 Royals also won the 1985 American League Championship Series in seven games.
  Fact 0.87: 1985 Royals became the first team to win the World Series after losing the first two games at home.
  Fact 0.88: 1985 Royals were the last team to come back from 3-1 down to win a title in any of the five major men's professional sports leagues until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3-1.
  Fact 0.89: 1985 Royals were the last team to win a World Series after trailing 3-1 until the Cubs did so in 2016.
  Fact 0.90: 1986 World Series: New York Mets beat Boston Red Sox 8-5 at Shea Stadium, Flushing, Queens, New York.
  Fact 0.91: 1986 World Series: The Mets became the first team to be one strike away from elimination and come back to win the series.
  Fact 0.92: 1986 World Series: The Mets became the second team to win the series after losing the first two games at home.
  Fact 0.93: A row with no year listed: Minnesota Twins beat St. Louis Cardinals 4-2 at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.94: A row with no year listed: The home team won all seven games of the series for the first time in MLB history.
  Fact 0.95: 1991 World Series: Minnesota Twins beat Atlanta Braves 1-0 in 10 innings (F/10) at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.96: 1991 World Series: Like in 1987, the home team won all games of the series.
  Fact 0.97: 1991 World Series: The Twins won in the bottom of the 10th on a walk-off single by Gene Larkin scoring Dan Gladden after 9 and a fraction scoreless innings of play.
  Fact 0.98: 1991 World Series: Jack Morris went all ten innings for the Twins and was named Series MVP.
  Fact 0.99: 1991 World Series: Twins centerfielder Kirby Puckett hit a famous walk-off home run in the bottom of the eleventh inning of the previous night's game to force this decisive game.
  Fact 0.100: 1997 World Series: Florida Marlins beat Cleveland Indians 3-2 in 11 innings (F/11) at Pro Player Stadium, Miami Gardens, Florida.
  Fact 0.101: 1997 World Series: Florida became the first wild card team to win the World Series.
  Fact 0.102: 1997 World Series: The Marlins rallied for one run in the bottom of the ninth to tie.
  Fact 0.103: 1997 World Series: The Marlins won in the eleventh on an Édgar Rentería RBI single off Charles Nagy.
  Fact 0.104: 2001 World Series: Arizona Diamondbacks beat New York Yankees 3-2 at Bank One Ballpark, Phoenix, Arizona.
  Fact 0.105: 2001 World Series: Arizona became the fastest expansion team to win the World Series.
  Fact 0.106: 2001 World Series: Arizona rallied for two runs in the bottom of the ninth for a come-from-behind win.
  Fact 0.107: 2001 World Series: Luis Gonzalez hit the winning RBI single off closer Mariano Rivera.
  Fact 0.108: 2001 World Series: The home team won every game of the series.
  Fact 0.109: 2002 World Series: Anaheim Angels beat San Francisco Giants 4-1 at Edison International Field, Anaheim, California.
  Fact 0.110: 2002 World Series: Both World Series teams were wild card entrants into the playoffs.
  Fact 0.111: 2002 World Series: John Lackey became the second rookie pitcher to win a World Series game seven.
  Fact 0.112: 2011 World Series: St. Louis Cardinals beat Texas Rangers 6-2 at Busch Stadium, St. Louis, Missouri.
  Fact 0.113: 2011 World Series: The Cardinals became the second team to be one strike away from elimination and come back to win the series.
  Fact 0.114: 2011 World Series: The Cardinals did come back to win the series twice.
  Fact 0.115: 2011 World Series: Just like the 1991 series, a walk-off home run in the bottom of the eleventh inning of game six forced this decisive game.
  Fact 0.116: 2011 World Series: David Freese provided the firepower in that game six walk-off home run.
  Fact 0.117: 2011 World Series: The most recent game seven was won by the home team.
  Fact 0.118: 2014 World Series: San Francisco Giants beat Kansas City Royals 3-2 at Kauffman Stadium, Kansas City, Missouri.
  Fact 0.119: 2014 World Series: Like in 2002, both World Series teams were wild card entrants into the playoffs.
  Fact 0.120: 2014 World Series: The Giants became the first NL team to win a game seven on the road since 1979.
  Fact 0.121: 2014 World Series: Madison Bumgarner was the NLCS MVP and eventual World Series MVP.
  Fact 0.122: 2014 World Series: Madison Bumgarner is normally a starter.
  Fact 0.123: 2014 World Series: Madison Bumgarner pitched five scoreless innings of relief on two days rest.
  Fact 0.124: 2014 World Series: Madison Bumgarner's relief pitching saved the Giants' third title in five years.
  Fact 0.125: 2016 World Series: Chicago Cubs beat Cleveland Indians 8-7 in 10 innings (F/10) at Progressive Field, Cleveland, Ohio.
  Fact 0.126: 2016 World Series: It was the Cubs' first World Series win since 1908.
  Fact 0.127: 2016 World Series: The game was played between the two teams with the longest active championship droughts.
  Fact 0.128: 2016 World Series: The Cubs are the first Major League Baseball team to come back to win the World Series from a 3-1 series deficit since the 1985 Royals.
  Fact 0.129: 2016 World Series: It was the most recent World Series game seven to be decided in extra innings.
  Fact 0.130: 2016 World Series: It was the first extra innings game seven to be won by the road team.
  Fact 0.131: 2017 World Series: Houston Astros beat Los Angeles Dodgers 5-1 at Dodger Stadium, Los Angeles.
  Fact 0.132: 2017 World Series: It was the first instance of back-to-back game sevens since 2001-2002.
  Fact 0.133: 2017 World Series: It was the first World Series championship for the Astros.
  Fact 0.134: 2017 World Series: The Astros became the first AL team to win a World Series game 7 on the road since 1972.
  Fact 0.135: 2017 World Series: The Astros became the second team to win two winner-take-all game sevens in the same postseason since the 1985 Kansas City Royals.

Retrieved items:
<item 1>
Game seven
1926 St. Louis Cardinals New York Yankees 3 -- 2 Yankee Stadium , the Bronx , New York City Grover Cleveland Alexander strikes out Tony Lazzeri with the bases
loaded to end the seventh and preserve the Cardinals lead . Two innings later Babe Ruth is caught stealing to end the World Series . 1931 St. Louis Cardinals
Philadelphia Athletics 4 -- 2 St. Louis , Missouri Last postseason game for the Athletics while based in Philadelphia . 1934 St. Louis Cardinals Detroit Tigers
11 -- 0 Navin Field , Detroit Dizzy Dean 's second win , a six - hit shutout on one day of rest , gives the Cardinals the title . 1940 Cincinnati Reds Detroit
Tigers 2 -- 1 Crosley Field , Cincinnati This game was over in a snappy 1 : 47 while a sacrifice bunt and sacrifice fly scored the winning runs . 1945 Detroit
Tigers Chicago Cubs 9 -- 3 Wrigley Field , Chicago Cubs last World Series appearance until 2016 . See also : Curse of the Billy Goat 1946 St. Louis Cardinals
Boston Red Sox 4 -- 3 Sportsman 's Park III , St.
</item 1>

<item 2>
Game seven
2003 NLCS Florida Marlins Chicago Cubs 9 -- 6 Wrigley Field , Chicago One night after the Steve Bartman incident , the Marlins advance to the World Series ,
where they would win in six games , and the Cubs ' World Series drought continues . ALCS Boston Red Sox New York Yankees 10 -- 3 Yankee Stadium , the Bronx ,
New York City Boston rallied from an 0 -- 3 deficit to win the American League pennant , the only time in Major League Baseball history that this has happened
in a best of seven series . Boston would go on to win the World Series for the first time since 1918 . NLCS St. Louis Cardinals Houston Astros 5 -- 2 Busch
Stadium ( II ) , St. Louis , Missouri The home team won all games of the series . Last time both LCS went to a game seven in the same year . Scott Rolen 's two
- run homer off Roger Clemens in the sixth inning puts the Cardinals in front for good . This is the last win of the season for St. Louis as the Cardinals got
steamrolled by Boston in four games . 2006 NLCS St.
</item 2>

<item 3>
Game seven
1 World Series 1.1. 2 League Championship Series 1.2 National Hockey League 1.3 National Basketball Association 1.4 Chinese Professional Baseball League 2
Comebacks 3 References Examples ( edit ) Major League Baseball ( edit ) In the MLB Postseason , game seven can occur in the League Championship Series and the
World Series ( the divisional series are best of five ) . In the World Series , there have been 39 decisive game sevens ( of the four World Series that were
contested on a best - of - nine basis , none went to a game nine , though all four included a non-decisive game seven ) . World Series ( edit ) World Series
that were decided on a game seven : All extra inning game sevens involve three franchises each with two appearances , the New York / San Francisco Giants (
losing both ) , the Washington Senators / Minnesota Twins ( winning both ) and the Cleveland Indians ( losing both ) . Home teams are 19 -- 20 in game sevens .
</item 3>

<item 4>
2017 World Series
Game 5 lasted five hours and seventeen minutes , making it the second longest World Series game in history by time , trailing only the 14 inning contest between
the Astros and the Chicago White Sox in Game 3 of the 2005 World Series in the same ballpark , as well as the third World Series game ever to go beyond five
hours of game time . This game is considered one of the greatest World Series games of all time . Game 6 ( edit ) Kenley Jansen recorded a two - inning save in
Game 6 .
</item 4>

<item 5>
Game seven
1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999
2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most
Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV
ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World
Series Stanley Cup Finals Stanley Cup awarded on challenge basis 1893 --
</item 5>

<item 6>
Game seven
Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League
Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in
extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First
instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World
Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City
Royals in 1985 . League Championship Series ( edit ) The League Championship Series were originally decided best of five from 1969 until 1984 . Since 1985 when
both were expanded to best of seven contests , 16 series ( 7 American , 9 National ) were decided in game seven .
</item 6>

<item 7>
2017 World Series
This was the first time since the 2001 World Series and 2002 World Series that back - to - back Fall Classics had a Game 7 . The starting pitchers for this game
were the same as in the third game : McCullers and Darvish . Springer doubled to open the game and scored the first run on an error by Bellinger , which allowed
Bregman to reach second base . Bregman stole third base and scored on an Altuve groundout . McCann scored the next inning on a groundout hit by the pitcher
McCullers . Springer hit a two - run home run , his fifth of the series , tying Reggie Jackson and Chase Utley for most home runs in a single World Series and
setting a new record with 29 total bases in any postseason series . As in Game 3 , Darvish lasted only ​ 1 ⁄ innings ( tying the shortest outing of his career )
and became the third pitcher with two starts of less than two innings in a World Series , and the first since Art Ditmar in the 1960 World Series .
</item 7>

<item 8>
Game seven
2012 NLCS San Francisco Giants St. Louis Cardinals 9 -- 0 AT&T Park , San Francisco , California First Giants victory in a winner - take - all game seven . The
Giants would go on to sweep the Detroit Tigers in the World Series . 2017 ALCS Houston Astros New York Yankees 4 -- 0 Minute Maid Park , Houston , Texas The
home team won all games of the series . Houston became the first franchise to win the pennant in both the National League and American League . The Astros would
go on to win the World Series over the Los Angeles Dodgers in seven games , becoming the second MLB team to win two game sevens in a single postseason .
National Hockey League ( edit ) Main article : List of NHL game sevens In the NHL playoffs , game seven can occur in all playoff series . In the Stanley Cup
Final , the NHL gained complete control of the Cup in 1926 , and the league instituted the best - of - seven series starting in the 1938 -- 39 season . No
Stanley Cup Final game seven has ever ended with a 1 -- 0 score .
</item 8>

<item 9>
Game seven
Louis the NL Pennant . The Cardinals would lose the World Series as the Minnesota Twins give new meaning to the term home - field advantage in the World Series
as only the decisive game seven is close . 1988 NLCS Los Angeles Dodgers New York Mets 6 -- 0 Dodger Stadium , Los Angeles Orel Hershiser pitches a five - hit
shutout to send the Dodgers to the World Series . Until 2017 , this was the only game seven of any postseason series to be played in Dodger Stadium . 1991 NLCS
Atlanta Braves Pittsburgh Pirates 4 -- 0 Three Rivers Stadium , Pittsburgh , Pennsylvania Back - to - back shutouts in games six and seven helps the Braves win
the pennant . The Braves would lose the series to Minnesota in seven games in what is regarded as the best World Series of all time . 1992 NLCS Atlanta Braves
Pittsburgh Pirates 3 -- 2 Atlanta -- Fulton County Stadium , Atlanta Atlanta scored all three runs in the bottom of the ninth . Pinch hitter Francisco Cabrera
drove in the two winning runs with a two - out single with the bases loaded . The winning run was scored by Sid Bream .
</item 9>

<item 10>
Game seven
League Championship Series that were decided on a game seven : Year Series Winning team Losing team Score Site Notes 1985 ALCS Kansas City Royals Toronto Blue
Jays 6 -- 2 Exhibition Stadium , Toronto The Royals rallied from a 3 - 1 deficit , including taking the last two on the road to win the pennant . They would go
on to win the 1985 World Series in seven games , becoming the first team in MLB history to win two game sevens in a single postseason . 1986 ALCS Boston Red Sox
California Angels 8 -- 1 Fenway Park , Boston , Massachusetts Dave Henderson 's famous home run off of Donnie Moore turned the tide as Boston rallied from a 3 -
1 deficit to win the series . Boston would lose the 1986 World Series in seven games . NLCS St. Louis Cardinals San Francisco Giants 6 -- 0 Busch Stadium ( II )
, St. Louis , Missouri Although Jeffrey Leonard would homer in the first four games of the series it was not enough as back - to - back shutouts by the
Cardinals in games six and seven gave St. Louis the NL Pennant .
</item 10>
=== END ITEM id=3d31a5406fa87e0a ===
