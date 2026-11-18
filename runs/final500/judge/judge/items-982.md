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

=== ITEM id=041bc66b3492e750 ===
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
Petroleum industry in Nigeria
The most productive region of the nation is the coastal Niger Delta Basin in the Niger Delta or `` South - south '' region which encompasses 78 of the 159 oil
fields . Most of Nigeria 's oil fields are small and scattered , and as of 1990 , these small unproductive fields accounted for 62.1 % of all Nigerian
production . This contrasts with the sixteen largest fields which produced 37.9 % of Nigeria 's petroleum at that time . As a result of the numerous small
fields , an extensive and well - developed pipeline network has been engineered to transport the crude oil . Also because of the lack of highly productive
fields , money from the jointly operated ( with the federal government ) companies is constantly directed towards petroleum exploration and production . Nigeria
's petroleum is classified mostly as `` light '' and `` sweet '' , as the oil is largely free of sulphur . Nigeria is the largest producer of sweet oil in OPEC
. This sweet oil is similar in composition to petroleum extracted from the North Sea . This crude oil is known as `` Bonny light '' .
</item 1>

<item 2>
Petroleum industry in Nigeria
Mobil operates primarily out of the Qua Iboe Terminal in Akwa Ibom State , while Chevron owns the Escravos Terminal located in Delta State and has a storage
capacity of 3.6 million barrels ( 570,000 m ) . Agip operates the Brass Terminal in Brass , a town 113 kilometres ( 70 miles ) southwest of Port Harcourt and
has a storage capacity of 3,558,000 barrels ( 565,700 m ) . Texaco operates the Pennington Terminal . Offshore ( edit ) Oil companies in Africa investigate
offshore production as an alternative area of production . Deepwater production mainly involves underwater drilling that exists 400 metres ( 1,300 ft ) or more
below the surface of the water . By expanding to deep water drilling the possible sources for finding new oil reserves is expanded . Through the introduction of
deep water drilling 50 % more oil is extracted than before the new forms of retrieving the oil . Angola and Nigeria are the largest oil producers in Africa . In
Nigeria , the deepwater sector still has a large avenue to expand and develop .
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
Nigeria, Africa’s largest oil producer and an OPEC member since 1971, depends heavily on petroleum: by 2000, oil and gas generated over 98% of export earnings,
83% of federal revenue, and 14% of GDP. Estimated reserves range from 16–22 billion barrels, with some estimates reaching 35.3 billion. Government officials
capture most profits, while citizens see few benefits and demand compensation. Refineries operate poorly or are defunct,
</item 4>

<item 5>
Petroleum industry in Nigeria
An open - air market for illegal crude oil operates off the Niger Delta , called the Togo Triangle . Natural gas ( edit ) Natural gas reserves are well over
5,300 km ( 187 × 10 ^ cu ft ) , the gas reserves are three times as substantial as the crude oil reserves . The biggest natural gas initiative is the Nigerian
Liquified Natural Gas Company , which is operated jointly by several companies and the state . It began exploration and production in 1999 . Chevron is also
attempting to create the Escravos Gas Utilization project which will be capable of producing 4,500,000 m ( 160 × 10 ^ cu ft ) per day . gas reserves . In 2008 ,
the government prepared a Gas Master Plan that was intended to promote natural gas production and encourage the supply of natural gas to domestic power stations
so as to help alleviate the country 's electricity shortages . There is also an export gas pipeline , known as the West African Gas Pipeline , in the works but
has encountered numerous setbacks . The pipeline would allow for transportation of natural gas to Benin , Ghana , Togo , and Cote d'Ivoire .
</item 5>

<item 6>
Sub-Saharan Africa
Transportation cost is low and no pipelines have to be laid as in Central Asia . Almost all reserves are offshore , so political turmoil within the host country
will not directly interfere with operations . Sub-Saharan oil is viscous , with a very low sulfur content . This quickens the refining process and effectively
reduces costs . New sources of oil are being located in Sub-Saharan Africa more frequently than anywhere else . Of all new sources of oil , 1⁄3 are in
Sub-Saharan Africa . Agriculture ( edit ) Agricultural fields in Rwanda 's Eastern Province Sub-Saharan Africa has more variety of grains than anywhere in the
world . Between 13,000 and 11,000 BCE wild grains began to be collected as a source of food in the cataract region of the Nile , south of Egypt . The collecting
of wild grains as source of food spread to Syria , parts of Turkey , and Iran by the eleventh millennium BCE . By the tenth and ninth millennia southwest Asians
domesticated their wild grains , wheat , and barley after the notion of collecting wild grains spread from the Nile . Numerous crops have been domesticated in
the region and spread to other parts of the world .
</item 6>

<item 7>
Sub-Saharan Africa
According to the World Bank , the economic growth rate in the region had risen to 4.7 % in 2013 , with a rate of 5.2 % forecasted for 2014 . This continued rise
was attributed to increasing investment in infrastructure and resources as well as steady expenditure per household . Energy and power ( edit ) Main article :
Mineral industry of Africa Oil production by country ( with other key actors of African or oil economy ) Rank Area bb / day Year Like ... _ W : World 85540000
2007 est . 01 E : Russia 9980000 2007 est . 02 Ar : Saudi Arb 9200000 2008 est . 04 As : Libya 4725000 2008 est . Iran 10 Af : Nigeria / Africa 2352000 2011 est
. Norway 15 Af : Algeria 2173000 2007 est . 16 Af : Angola 1910000 2008 est . 17 Af : Egypt 1845000 2007 est . 27 Af : Tunisia 664000 2007 est . Australia 31 Af
: Sudan 466100 2007 est .
</item 7>

<item 8>
Sub-Saharan Africa attracts major investment, including from China, and is rich in minerals: it produces about 30% of global gold, 33% of bauxite, 49% of
diamonds, and most platinum; South Africa dominates platinum and chromium, while Zambia leads copper and the DRC holds major coltan reserves. Niger, Namibia,
and South Africa supply uranium. Despite only 10% of proven oil reserves, the region draws intense global competition because offshore oil is relatively easy
and
</item 8>

<item 9>
Sub-Saharan Africa
Major suppliers are South Africa , Ghana , Zimbabwe , Tanzania , Guinea , and Mali . South Africa had been first in the world in terms of gold production since
1905 , but in 2007 it moved to second place , according to GFMS , the precious metals consultancy . Uranium is major commodity from the region . Significant
suppliers are Niger , Namibia , and South Africa . Namibia was the number one supplier from Sub-Saharan Africa in 2008 . The region produces 49 % of the world
's diamonds . By 2015 , it is estimated that 25 % of North American oil will be from Sub-Saharan Africa , ahead of the Middle East . Sub-Saharan Africa has been
the focus of an intense race for oil by the West , China , India , and other emerging economies , even though it holds only 10 % of proven oil reserves , less
than the Middle East . This race has been referred to as the second Scramble for Africa . All reasons for this global scramble come from the reserves ' economic
benefits . Transportation cost is low and no pipelines have to be laid as in Central Asia .
</item 9>

<item 10>
Sub-Saharan Africa
Barbary Coast Bashmur Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba
Mountains Nubia The Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf
of Guinea Niger Basin Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern
Highlands Rhodesia North South Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay
Hydra Bay Macro-regions Aethiopia Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater
Middle East Islands of Africa List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland
Northeast Africa Portuguese - speaking African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa
Regions of Asia Central Greater Middle East Aral Sea Aralkum Desert Caspian Sea Dead Sea Sea of Galilee Transoxiana Turan Greater Khorasan
</item 10>
=== END ITEM id=041bc66b3492e750 ===

=== ITEM id=375e74afef04ca77 ===
Gold sentences and facts:
Sentence 0: The season is scheduled to be released on March 8 , 2018 .
  Fact 0.0: The season is scheduled to be released on March 8, 2018.

Retrieved items:
<item 1>
Jessica Jones (season 2)
They are joined by Leah Gibson and J.R. Ramirez . The second season was ordered in January 2016 , with filming beginning in April 2017 , back - to - back with
the miniseries Marvel 's The Defenders . Filming concluded in September 2017 . The season is scheduled to be released on March 8 , 2018 . Contents ( hide ) 1
Cast 1.1 Main 1.2 Notable guests 2 Production 2.1 Development 2.2 Writing 2.3 Casting 2.4 Filming 3 Release 3.1 Marketing 4 References 5 External links Cast (
edit ) Further information : List of Jessica Jones characters Main ( edit ) Krysten Ritter as Jessica Jones Rachael Taylor as Patricia `` Trish '' Walker Carrie
- Anne Moss as Jeri Hogarth Eka Darville as Malcolm Ducasse Leah Gibson as Ingrid J.R.
</item 1>

<item 2>
Jessica Jones (season 2)
Deadline.com . Archived from the original on November 19 , 2015 . Retrieved November 18 , 2015 . ^ Jump up to : Ritter , Stephanie ( May 12 , 2016 ) . Jessica
Jones Season Two Details Revealed ! ( Exclusive ! ) . Nerdist Industries . Retrieved May 14 , 2016 . Jump up ^ Schwartz , Terri ( January 17 , 2016 ) . `` How
Jessica Jones ' Final Triumph Is n't An Easy Resolution '' . IGN . Archived from the original on January 19 , 2016 . Retrieved January 19 , 2016 . Jump up ^
Goldman , Eric ( January 17 , 2016 ) . `` Jessica Jones : Rachael Taylor on Trish 's Potential Hellcat Future in Season 2 '' . IGN . Archived from the original
on January 18 , 2016 . Retrieved January 18 , 2016 . Jump up ^ Radish , Christina ( January 28 , 2016 ) .
</item 2>

<item 3>
Jessica Jones (season 2)
Retrieved August 14 , 2017 . Jump up ^ Schwartz , Terri ( January 7 , 2015 ) . `` Netflix announces ' Daredevil , ' ' Bloodline , ' ' Kimmy Schmidt ' premiere
dates '' . Zap2It . Archived from the original on January 10 , 2015 . Retrieved January 10 , 2015 . Jump up ^ Goldman , Eric ( July 28 , 2015 ) . `` Netflix On
Marvel Series Release Plan And If Punisher Could Get A Spinoff '' . IGN . Archived from the original on July 28 , 2015 . Retrieved July 28 , 2015 . Jump up ^
Weintraub , Steve ( October 12 , 2015 ) . `` Krysten Ritter & Showrunner Melissa Rosenberg Talk ' Jessica Jones ' at NYCC '' . Collider . Archived from the
original on October 14 , 2015 . Retrieved October 15 , 2015 . Jump up ^ Ryan , Maureen ( November 20 , 2015 ) .
</item 3>

<item 4>
Jessica Jones (season 2)
Jump up ^ Ryan , Maureen ( November 20 , 2015 ) . `` ' Jessica Jones ' Showrunner Melissa Rosenberg Talks About Her Tough Heroine '' . Variety . Archived from
the original on November 21 , 2015 . Retrieved November 21 , 2015 . Jump up ^ Fitzpatrick , Kevin ( July 27 , 2016 ) . `` New ' Jessica Jones , ' ' Daredevil '
and ' Punisher ' No Earlier Than 2018 ? '' . ScreenCrush . Archived from the original on July 29 , 2016 . Retrieved July 28 , 2016 . Jump up ^ Friedlander ,
Whitney ( January 17 , 2016 ) . `` ' Jessica Jones ' Renewed for Season 2 on Netflix '' . Variety . Archived from the original on January 17 , 2016 . Retrieved
January 18 , 2016 . Jump up ^ Andreeva , Nellie ( March 31 , 2016 ) . `` ' Jessica Jones ' : Raelle Tucker Joins Netflix Series As Executive Producer '' .
</item 4>

<item 5>
Jessica Jones (season 2)
`` JESSICA JONES Renewed For Second Season '' . Newsarama . Archived from the original on January 18 , 2016 . Retrieved January 18 , 2016 . ^ Jump up to : Beard
, Jim ( February 3 , 2016 ) . `` Rachael Taylor Deals with Trish Walker 's Inner Demons in ' Marvel 's Jessica Jones ' '' . Marvel.com . Archived from the
original on February 5 , 2016 . Retrieved February 5 , 2016 . ^ Jump up to : Beard , Jim ( February 5 , 2016 ) . `` Carrie - Anne Moss Pays for Hogarth 's Sins
in ' Marvel 's Jessica Jones ' '' . Marvel.com . Archived from the original on February 6 , 2016 . Retrieved February 6 , 2016 . ^ Jump up to : Perry , Spencer
( August 17 , 2017 ) . `` Jessica Jones Season 2 Set Photos Tease How Kilgrave Returns '' . ComingSoon.net .
</item 5>

<item 6>
Jessica Jones (season 2)
Retrieved September 7 , 2016 . External links ( edit ) Marvel 's Jessica Jones on IMDb Marvel 's Jessica Jones at TV.com Jessica Jones Brian Michael Bendis
Michael Gaydos Supporting characters Luke Cage Carol Danvers Jessica Drew Ben Urich Enemies Man Mountain Marko Purple Man Publications Alias The Pulse Jessica
Jones In other media Jessica Jones season 1 characters Related articles Daily Bugle New Avengers Marvel Cinematic Universe Feature films Phase One Iron Man The
Incredible Hulk Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The
Winter Soldier Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2
Spider - Man : Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television
series ABC Agents of S.H.I.E.L.D.
</item 6>

<item 7>
Jessica Jones (season 2)
Jessica Jones ( season 2 ) - wikipedia Jessica Jones ( season 2 ) Jump to : navigation , search Jessica Jones ( season 2 ) Promotional poster Starring Krysten
Ritter Rachael Taylor Carrie - Anne Moss Eka Darville Leah Gibson J.R. Ramirez Country of origin United States Release Original network Netflix Season
chronology ← Previous Season 1 List of Jessica Jones episodes The second season of the American web television series Jessica Jones , which is based on the
Marvel Comics character of the same name , follows Jones as she takes on a new case after the events surrounding her encounter with Kilgrave . It is set in the
Marvel Cinematic Universe ( MCU ) , sharing continuity with the films and other television series of the franchise . The season will be produced by Marvel
Television in association with ABC Studios and Tall Girls Productions , with Melissa Rosenberg serving as showrunner . Krysten Ritter stars as Jones , with
Rachael Taylor , Carrie - Anne Moss , and Eka Darville also returning from the first season . They are joined by Leah Gibson and J.R. Ramirez .
</item 7>

<item 8>
Jessica Jones (season 2)
On January 17 , 2016 , Netflix ordered a second season of 13 episodes . Raelle Tucker joined the season as an executive producer and writer , replacing Liz
Friedman from the first season , who departed the series to work on the pilot for the ABC series , Conviction . Writing ( edit ) Rosenberg and the season 's
writers were halfway through the writing process by August 2016 , with the scripts completed by the end of October 2016 . With The Defenders releasing before
the season , Rosenberg used the miniseries as an opportunity to help `` set up '' elements for the season , working with The Defenders executive producers and
writers Doug Petrie and Marco Ramirez to do so . Rosenberg wanted to `` continue with ( the Jessica Jones ) character '' in the season , saying , `` She 's a
very damaged character , her damage goes beyond ( David Tennant 's ) Kilgrave . There 's a lot to mine from in her backstory and in her present day situation ''
.
</item 8>

<item 9>
Jessica Jones (season 2)
`` Netflix hit ' Jessica Jones ' films at Long Island Aquarium '' . Northforker . Archived from the original on June 12 , 2017 . Retrieved June 12 , 2017 . Jump
up ^ Ridgely , Charlie ( September 15 , 2017 ) . `` Filming Wraps on ' Jessica Jones ' Season 2 '' . Comicbook.com . Archived from the original on September 15
, 2017 . Retrieved September 15 , 2017 . ^ Jump up to : Ramos , Dino - Ray ( December 9 , 2017 ) . `` ' Marvel 's Jessica Jones ' : Season 2 Premiere Date Set
For Netflix Superhero Series '' . Deadline.com . Archived from the original on December 10 , 2017 . Retrieved December 10 , 2017 . Jump up ^ `` Marvel 's
Jessica Jones '' . Netflix . Archived from the original on September 7 , 2016 . Retrieved September 7 , 2016 .
</item 9>

<item 10>
Jessica Jones (season 2)
This was a goal that `` Marvel was completely on board with '' , and given the demand of many talented female directors at the time , the series ' producers
looked to book only female directors first , and approach male directors later in the pre-production phase if needed . Another member of the production
suggested that the series book only female directors for the season , which Rosenberg `` had n't contemplated ( as a ) concept prior to that conversation '' .
She quickly made that the goal of the production , and in October 2016 , Rosenberg confirmed that all 13 episodes of the season would be directed by women .
Release ( edit ) The second season of Jessica Jones is scheduled to be released on March 8 , 2018 , on the streaming service Netflix worldwide , in Ultra HD 4K
and high dynamic range . Marketing ( edit ) In December 2017 , a teaser trailer for the season was released , along with announcing the season release date .
References ( edit ) ^ Jump up to : Arrant , Chris ( January 17 , 2016 ) . `` JESSICA JONES Renewed For Second Season '' . Newsarama .
</item 10>
=== END ITEM id=375e74afef04ca77 ===

=== ITEM id=8ba10111e84b88fc ===
Gold sentences and facts:
Sentence 0: As a technique for drawing , the closest predecesor to the pencil was Silverpoint until in 1565 ( some sources say as early as 1500 ) , a large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite in Borrowdale parish , Cumbria , England . This particular deposit of graphite was extremely pure and solid , and it could easily be sawn into sticks . It remains the only large - scale deposit of graphite ever found in this solid form . Chemistry was in its infancy and the substance was thought to be a form of lead . Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally mean lead pen .
  Fact 0.0: Before the pencil, the closest predecessor as a drawing technique was Silverpoint, until 1565 (some sources say as early as 1500).
  Fact 0.1: A large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite.
  Fact 0.2: The Grey Knotts graphite deposit is in Borrowdale parish, Cumbria, England.
  Fact 0.3: The graphite deposit at Grey Knotts was extremely pure and solid.
  Fact 0.4: The graphite deposit at Grey Knotts could easily be sawn into sticks.
  Fact 0.5: The Grey Knotts graphite deposit remains the only large-scale deposit of graphite ever found in this solid form.
  Fact 0.6: Chemistry was in its infancy when the graphite was found.
  Fact 0.7: The substance was thought to be a form of lead.
  Fact 0.8: The substance was called plumbago, Latin for 'lead ore'.
  Fact 0.9: The pencil core is still referred to as 'lead' or 'a lead'.
  Fact 0.10: Many people have the misconception that the graphite in the pencil is lead.
  Fact 0.11: The black core of pencils is still referred to as lead.
  Fact 0.12: The black core of pencils never contained the element lead.
  Fact 0.13: The German word for pencil, bleistift, literally means lead pen.
  Fact 0.14: The Irish word for pencil, peann luaidhe, literally means lead pen.
  Fact 0.15: The Arabic word for pencil, قلم رصاص (qalam raṣāṣ), literally means lead pen.
  Fact 0.16: Some other languages also use words for pencil that literally mean lead pen.

Retrieved items:
<item 1>
Pencil
Their concept involved the hollowing out of a stick of juniper wood . Shortly thereafter , a superior technique was discovered : two wooden halves were carved ,
a graphite stick inserted , and the halves then glued together -- essentially the same method in use to this day . New pencils from graphite powder , and
graphite and clay The first attempt to manufacture graphite sticks from powdered graphite was in Nuremberg , Germany , in 1662 . It used a mixture of graphite ,
sulphur , and antimony . English and German pencils were not available to the French during the Napoleonic Wars ; France , under naval blockade imposed by Great
Britain , was unable to import the pure graphite sticks from the British Grey Knotts mines -- the only known source in the world . France was also unable to
import the inferior German graphite pencil substitute . It took the efforts of an officer in Napoleon 's army to change this . In 1795 , Nicolas - Jacques Conté
discovered a method of mixing powdered graphite with clay and forming the mixture into rods that were then fired in a kiln . By varying the ratio of graphite to
clay , the hardness of the graphite rod could also be varied .
</item 1>

<item 2>
Pencil
Contents ( hide ) 1 History 1.1 Camel 's hair pencil 1.2 Discovery of graphite deposit 1.3 Wood holders added 1.4 New pencils from graphite powder , and
graphite and clay 1.5 The pencil in America 1.6 Eraser attached 1.7 Pencil extenders 2 Types 2.1 By marking material 2.2 By use 2.3 By shape 2.4 By size 2.5 By
manufacture 3 Health 4 Manufacture 5 Grading and classification 5.1 Pencil hardness test 6 External colour and shape of pencils 7 Notable pencil users 8
Manufacturers 9 See also 10 References 11 Bibliography 12 Further reading 13 External links History Old Soviet colored pencils with box ( circa 1959 ) Camel 's
hair pencil Pencil , from Old French pincel , from Latin penicillus a `` little tail '' ( see penis ; pincellus is Latin from the post-classical period )
originally referred to an artist 's fine brush of camel hair , also used for writing before modern lead or chalk pencils .
</item 2>

<item 3>
Pencil
Because graphite is soft , it requires some form of encasement . Graphite sticks were initially wrapped in string or sheepskin for stability . England would
enjoy a monopoly on the production of pencils until a method of reconstituting the graphite powder was found in 1662 in Italy . However , the distinctively
square English pencils continued to be made with sticks cut from natural graphite into the 1860s . The town of Keswick , near the original findings of block
graphite , still manufactures pencils , the factory also being the location of the Cumberland Pencil Museum . The meaning of `` graphite writing implement ''
apparently evolved late in the 16th century . Wood holders added Palomino Blackwing 602 pencils Around 1560 , an Italian couple named Simonio and Lyndiana
Bernacotti made what are likely the first blueprints for the modern , wood - encased carpentry pencil . Their version was a flat , oval , more compact type of
pencil . Their concept involved the hollowing out of a stick of juniper wood .
</item 3>

<item 4>
Pencil
Archived from the original on 5 March 2016 . Retrieved 5 October 2015 . Jump up ^ Bennett , Howard J. ( 26 November 2014 ) . `` Ever wondered about the lead in
pencils ? '' . The Washington Post . ISSN 0190 - 8286 . Archived from the original on 6 November 2015 . Retrieved 5 October 2015 . Jump up ^ `` Pencil
swallowing : MedlinePlus Medical Encyclopedia '' . www.nlm.nih.gov . Archived from the original on 6 October 2015 . Retrieved 5 October 2015 . Jump up ^ ``
graphite pencils The Weekend Historian '' . umeshmadan.wordpress.com . Archived from the original on 7 October 2015 . Retrieved 5 October 2015 . Jump up ^ ``
Origins of the Pencil ( Circa 1500 -- 1565 ) : HistoryofInformation.com '' . www.historyofinformation.com . Retrieved 2016 - 10 - 29 .
</item 4>

<item 5>
Pencil
Though the archetypal pencil was an artist 's brush , the stylus , a thin metal stick used for scratching in papyrus or wax tablets , was used extensively by
the Romans and for palm - leaf manuscripts . Discovery of graphite deposit As a technique for drawing , the closest predecesor to the pencil was Silverpoint
until in 1565 ( some sources say as early as 1500 ) , a large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite in
Borrowdale parish , Cumbria , England . This particular deposit of graphite was extremely pure and solid , and it could easily be sawn into sticks . It remains
the only large - scale deposit of graphite ever found in this solid form . Chemistry was in its infancy and the substance was thought to be a form of lead .
Consequently , it was called plumbago ( Latin for `` lead ore '' ) .
</item 5>

<item 6>
Pencil
www.historyofinformation.com . Retrieved 2016 - 10 - 29 . Jump up ^ `` Lead Facts - Uses , Properties , Element Pb , Plumbing , Pipes , Weights '' .
www.sciencekids.co.nz . Archived from the original on 30 October 2016 . Retrieved 29 October 2016 . Jump up ^ `` Keswick Pencil Museum '' . Pencilmuseum.co.uk .
Archived from the original on 3 August 2009 . Retrieved 23 July 2009 . Jump up ^ Douglas Harper ( 27 June 2012 ) . `` pencil '' . Online Etymology Dictionary .
Archived from the original on 23 August 2012 . Retrieved 27 June 2012 . Jump up ^ `` Who invented the pencil ? '' . Rocket City Space Pioneers . Archived from
the original on 2011 - 10 - 22 . Jump up ^ `` Timeline Outline ViewEra : 1500 - 1550 '' . Historyofscience.com . Retrieved 18 August 2012 . Jump up ^ `` Pencil
'' .
</item 6>

<item 7>
Pencil
Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people
have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the
element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally
mean lead pen . The value of graphite would soon be realised to be enormous , mainly because it could be used to line the moulds for cannonballs ; the mines
were taken over by the Crown and were guarded . When sufficient stores of graphite had been accumulated , the mines were flooded to prevent theft until more was
required . The usefulness of graphite for pencils was discovered as well , but graphite for pencils had to be smuggled . The news of the usefulness of these
early pencils spread far and wide , attracting the attention of artists all over the known world . Because graphite is soft , it requires some form of
encasement .
</item 7>

<item 8>
Pencil
Manufacture The lead of the pencil is a mix of finely ground graphite and clay powders . Before the two substances are mixed , they are separately cleaned of
foreign matter and dried in a manner that creates large square cakes . Once the cakes have fully dried , the graphite and the clay squares are mixed together
using water . The amount of clay content added to the graphite depends on the intended pencil hardness ( lower proportions of clay makes the core softer ) , and
the amount of time spent on grinding the mixture determines the quality of the lead . The mixture is then shaped into long spaghetti - like strings ,
straightened , dried , cut , and then tempered in a kiln . The resulting strings are dipped in oil or molten wax , which seeps into the tiny holes of the
material and allows for the smooth writing ability of the pencil . A juniper or incense - cedar plank with several long parallel grooves is cut to fashion a ``
slat , '' and the graphite / clay strings are inserted into the grooves . Another grooved plank is glued on top , and the whole assembly is then cut into
individual pencils , which are then varnished or painted .
</item 8>

<item 9>
Pencil
This method of manufacture , which had been earlier discovered by the Austrian Joseph Hardtmuth , the founder of the Koh - I - Noor in 1790 , remains in use .
In 1802 , the production of graphite leads from graphite and clay was patented by the Koh - I - Noor company in Vienna . In England , pencils continued to be
made from whole sawn graphite . Henry Bessemer 's first successful invention ( 1838 ) was a method of compressing graphite powder into solid graphite thus
allowing the waste from sawing to be reused . The pencil in america Pencil , perhaps made by Henry David Thoreau , in the Concord Museum Pencil manufacturing .
The top sequence shows the old method that required pieces of graphite to be cut to size ; the lower sequence is the new , current method using rods of graphite
and clay . American colonists imported pencils from Europe until after the American Revolution . Benjamin Franklin advertised pencils for sale in his
Pennsylvania Gazette in 1729 , and George Washington used a three - inch pencil when he surveyed the Ohio Territory in 1762 .
</item 9>

<item 10>
Pencil
Retrieved 23 July 2009 . Jump up ^ `` Pencils , paint and pottery can give you lead poisoning '' . Life . 73 ( 1 ) : 46 . 7 July 1972 . Jump up ^ `` The HB
Graphite Grading Scale '' . Pencils.com . Archived from the original on 21 March 2012 . Retrieved 14 March 2012 . Jump up ^ Petroski , 1990 , `` Appendix A Jump
up ^ `` Staedtler Pencil , video '' . Archived from the original on 21 January 2013 . Retrieved 23 July 2009 . Staedtler.com Jump up ^ Petroski , 1990 Jump up ^
`` Derwent Manufacturing Process '' . Pencils.co.uk . Archived from the original on 17 May 2013 . Retrieved 23 July 2009 . Jump up ^ `` Archived copy '' .
Archived from the original on 6 February 2017 . Retrieved 5 February 2017 .
</item 10>
=== END ITEM id=8ba10111e84b88fc ===

=== ITEM id=70b7e9907ed3fc9d ===
Gold sentences and facts:
Sentence 0: Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio .
  Fact 0.0: Much of the humus in most soils has persisted for more than 100 years rather than having been decomposed into CO.
  Fact 0.1: The long-persisting humus in most soils can be regarded as stable.
  Fact 0.2: Much of the humus in most soils has been protected from decomposition by microbial or enzyme action.
  Fact 0.3: Protected humus is hidden (occluded) inside small aggregates of soil particles, or is tightly sorbed or complexed to clays.
  Fact 0.4: Most humus that is not protected in this way is decomposed within 10 years.
  Fact 0.5: Humus that is not protected in this way can be regarded as less stable or more labile.
  Fact 0.6: Stable humus contributes few plant-available nutrients in soil.
  Fact 0.7: Stable humus helps maintain the physical structure of soil.
  Fact 0.8: A very stable form of humus is formed from the slow oxidation of soil carbon after finely powdered charcoal is incorporated into the topsoil.
  Fact 0.9: The formation of the very stable humus from charcoal is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio.

Retrieved items:
<item 1>
Humus
, 2011 . Accessed 24 Nov 2011 . < http://www.britannica.com/EBchecked/topic/276408/humus > . Jump up ^ Chertov , O.G. ; Kornarov , A.S. ; Crocker , G. ; Grace ,
P. ; Klir , J. ; Körschens , M. ; Poulton , P.R. ; Richter , D. ( 1997 ) . `` Simulating trends of soil organic carbon in seven long - term experiments using
the SOMM model of the humus types '' . Geoderma. 81 : 121 -- 135 . doi : 10.1016 / S0016 - 7061 ( 97 ) 00085 - 2 . Jump up ^ Baritz , R. , 2003 . Humus forms in
forests of the northern German lowlands . Schweizerbart , Stuttgart , Germany , 145 pp. ( 2 ) Jump up ^ Bunting , B.T. ; Lundberg , J. ( 1995 ) .
</item 1>

<item 2>
Humus
Most soils have three major horizons : the surface horizon ( A ) , the subsoil ( B ) , and the substratum ( C ) . Some soils have an organic horizon ( O ) on
the surface , but this horizon can also be buried . The master horizon ( E ) is used for subsurface horizons that have significantly lost minerals ( eluviation
) . Bedrock , which is not soil , uses the letter R . Benefits of soil organic matter and humus ( edit ) The importance of chemically stable humus is thought by
some to be the fertility it provides to soils in both a physical and chemical sense , though some agricultural experts put a greater focus on other features of
it , such as its ability to suppress disease . It helps the soil retain moisture by increasing microporosity , and encourages the formation of good soil
structure . The incorporation of oxygen into large organic molecular assemblages generates many active , negatively charged sites that bind to positively
charged ions ( cations ) of plant nutrients , making them more available to the plant by way of ion exchange . Humus allows soil organisms to feed and reproduce
, and is often described as the `` life - force '' of the soil .
</item 2>

<item 3>
Humus
The process that converts soil organic matter into humus feeds the population of microorganisms and other creatures in the soil , and thus maintains high and
healthy levels of soil life . The rate at which soil organic matter is converted into humus promotes ( when fast ) or limits ( when slow ) the coexistence of
plants , animals , and microorganisms in the soil . Effective humus and stable humus are additional sources of nutrients for microbes : the former provides a
readily available supply and the latter acts as a longeval storage reservoir . Decomposition of dead plant material causes complex organic compounds to be
slowly oxidized ( lignin - like humus ) or to decompose into simpler forms ( sugars and amino sugars , and aliphatic and phenolic organic acids ) , which are
further transformed into microbial biomass ( microbial humus ) or reorganized , and further oxidized , into humic assemblages ( fulvic and humic acids ) , which
bind to clay minerals and metal hydroxides . The ability of plants to absorb humic substances with their roots and metabolize them has been long debated . There
is now a consensus that humus functions hormonally rather than simply nutritionally in plant physiology .
</item 3>

<item 4>
Humus
Stability ( edit ) Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as
stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of
soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as
less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable
form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated
to have been important in the formation of the very fertile Amazonian terra preta do Indio . Horizons ( edit ) Humus has a characteristic black or dark brown
color and is organic due to an accumulation of organic carbon . Soil scientists use the capital letters O , A , B , C , and E to identify the master horizons ,
and lowercase letters for distinctions of these horizons .
</item 4>

<item 5>
Humus
Humus has a characteristic black or dark brown color and is an accumulation of organic carbon . Besides the three major soil horizons of ( A ) surface / topsoil
, ( B ) subsoil , and ( C ) substratum , some soils have an organic horizon ( O ) on the very surface . Hard bedrock ( R ) is not in a strict sense soil . In
soil science , humus ( derived in 1790 -- 1800 from the Latin humus for earth , ground ) denominates the fraction of soil organic matter that is amorphous and
without the `` cellular cake structure characteristic of plants , micro-organisms or animals . '' Humus significantly affects the bulk density of soil and
contributes to its retention of moisture and nutrients . In agriculture , `` humus '' sometimes also is used to describe mature or natural compost extracted
from a woodland or other spontaneous source for use as a soil conditioner . It is also used to describe a topsoil horizon that contains organic matter ( humus
type , humus form , humus profile ) . Humus is the dark organic matter that forms in soil when dead plant and animal matter decays .
</item 5>

<item 6>
Humus
Humus has many nutrients that improve the health of soil , nitrogen being the most important . The ratio of carbon to nitrogen ( C : N ) of humus is 10 : 1 .
Contents ( hide ) 1 Description 2 Humification 3 Stability 4 Horizons 5 Benefits of soil organic matter and humus 6 See also 7 References 8 External links
Description ( edit ) It is difficult to define humus precisely because it is a very complex substance which is not fully understood . Humus is different from
decomposing soil organic matter . The latter looks rough and has visible remains of the original plant or animal matter . Fully humified humus , on the contrary
, has a uniformly dark , spongy , and jelly - like appearance , and is amorphous ; it may gradually decompose over several years or persist for millennia . It
has no determinate shape , structure , or quality . However , when examined under a microscope , humus may reveal tiny plant , animal , or microbial remains
that have been mechanically , but not chemically , degraded . This suggests an ambiguous boundary between humus and soil organic matter .
</item 6>

<item 7>
Humus
This suggests an ambiguous boundary between humus and soil organic matter . While distinct , humus is an integral part of soil organic matter . Humification (
edit ) Microorganisms decompose a large portion of the soil organic matter into inorganic minerals that the roots of plants can absorb as nutrients . This
process is termed `` mineralization '' . In this process , nitrogen ( nitrogen cycle ) and the other nutrients ( nutrient cycle ) in the decomposed organic
matter are recycled . Depending on the conditions in which the decomposition occurs , a fraction of the organic matter does not mineralize , and instead is
transformed by a process called `` humification '' into concatenations of organic polymers . Because these organic polymers are resistant to the action of
microorganisms , they are stable , and constitute humus . This stability implies that humus integrates into the permanent structure of the soil , thereby
improving it . Humification can occur naturally in soil or artificially in the production of compost . Organic matter is humified by a combination of
saprotrophic fungi , bacteria , microbes and animals such as earthworms , nematodes , protozoa , and arthropods .
</item 7>

<item 8>
Humus
Soil Biology and Biochemistry . 26 ( 2 ) : 183 -- 220 . doi : 10.1016 / 0038 - 0717 ( 94 ) 90161 - 9 . Jump up ^ Humintech ® Definition Of Soil Organic Matter &
Humic Acids Based Products Jump up ^ Brady , N.C. ; Weil , R.R. ( 1999 ) . The nature and properties of soils . Upper Saddle River , N.J. : Prentice Hall , Inc
. Jump up ^ Soil biology Jump up ^ Berg , B. , McClaugherty , C. , 2007 . Plant litter : decomposition , humus formation , carbon sequestration , 2nd ed .
Springer , 338 pp. , ISBN 3 - 540 - 74922 - 5 Jump up ^ Levin , L. ; Forchiassin , F. ; Ramos , A.M. ( 2002 ) . `` Copper induction of lignin - modifying
enzymes in the white - rot fungus Trametes trogii '' . Mycologia. 94 : 377 -- 383 .
</item 8>

<item 9>
Humus
Plant remains , including those that animals digested and excreted , contain organic compounds : sugars , starches , proteins , carbohydrates , lignins , waxes
, resins , and organic acids . Decay in the soil begins with the decomposition of sugars and starches from carbohydrates , which decompose easily as
detritivores initially invade the dead plant organs , while the remaining cellulose and lignin decompose more slowly . Simple proteins , organic acids ,
starches , and sugars decompose rapidly , while crude proteins , fats , waxes , and resins remain relatively unchanged for longer periods of time . Lignin ,
which is quickly transformed by white - rot fungi , is one of the primary precursors of humus , together with by - products of microbial and animal activity .
The humus produced by humification is thus a mixture of compounds and complex biological chemicals of plant , animal , or microbial origin that has many
functions and benefits in soil . Some judge earthworm humus ( vermicompost ) to be the optimal organic manure .
</item 9>

<item 10>
Humus
Humus is a colloidal substance and increases the cation exchange capacity of soil , hence its ability to store nutrients by chelation . While these nutrient
cations are available to plants , they are held in the soil and prevented from being leached by rain or irrigation . Humus can hold the equivalent of 80 -- 90 %
of its weight in moisture , and therefore increases the soil 's capacity to withstand drought . The biochemical structure of humus enables it to moderate , i.e.
buffer , excessive acidic or alkaline soil conditions . During humification , microbes secrete sticky , gum - like mucilages ; these contribute to the crumby
structure ( tilth ) of the soil by adhering particles together and allowing greater aeration of the soil . Toxic substances such as heavy metals and excess
nutrients can be chelated , i.e. , bound to the organic molecules of humus , and so prevented from leaching away . The dark , usually brown or black , color of
humus helps to warm cold soils in Spring .
</item 10>
=== END ITEM id=70b7e9907ed3fc9d ===

=== ITEM id=ceb18ff6eb57f363 ===
Gold sentences and facts:
Sentence 0: Your Lie in April Cover of the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono 四 月 は 君 の 嘘 ( Shigatsu wa Kimi no Uso ) Manga Written by Naoshi Arakawa Published by Kodansha English publisher Kodansha Comics Demographic Shōnen Magazine Monthly Shōnen Magazine Original run April 6 , 2011 -- February 6 , 2015 Volumes 11 ( List of volumes ) Anime television series Directed by Kyōhei Ishiguro Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1 Pictures Licensed by Madman Entertainment Aniplex of America Anime Limited Original network Fuji TV ( Noitamina ) Original run October 9 , 2014 -- March 19 , 2015 Episodes 22 ( List of episodes ) Light novel Your Lie in April - A Six Person Etude Written by Yui Tokiumi Published by Kodansha English publisher Vertical Published November 17 , 2014 Volumes Manga Shigatsu wa Kimi no Uso : Coda Written by Naoshi Arakawa Demographic Shōnen Original run February 25 , 2015 -- October 28 , 2015 Volumes Original video animation Moments Directed by Kazuya Iwata Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1 Pictures Released May 15 , 2015 Runtime 23 minutes Live - action film Your Lie in April ( film ) Anime and Manga portal
  Fact 0.0: The cover shows the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono.
  Fact 0.1: The title is 四 月 は 君 の 嘘 (Shigatsu wa Kimi no Uso).
  Fact 0.2: The manga Shigatsu wa Kimi no Uso is written by Naoshi Arakawa.
  Fact 0.3: The manga is published by Kodansha, with English publisher Kodansha Comics.
  Fact 0.4: The manga has the Shōnen demographic and was serialized in Monthly Shōnen Magazine.
  Fact 0.5: The manga's original run was April 6, 2011 to February 6, 2015.
  Fact 0.6: The manga has 11 volumes.
  Fact 0.7: The anime television series is directed by Kyōhei Ishiguro.
  Fact 0.8: The anime television series is written by Takao Yoshioka.
  Fact 0.9: The anime television series has music by Masaru Yokoyama.
  Fact 0.10: The anime television series is by studio A-1 Pictures.
  Fact 0.11: The anime is licensed by Madman Entertainment and Aniplex of America Anime Limited.
  Fact 0.12: The anime's original network is Fuji TV (Noitamina).
  Fact 0.13: The anime's original run was October 9, 2014 to March 19, 2015.
  Fact 0.14: The anime has 22 episodes.
  Fact 0.15: The light novel Your Lie in April - A Six Person Etude is written by Yui Tokiumi.
  Fact 0.16: The light novel is published by Kodansha, with English publisher Vertical.
  Fact 0.17: The light novel was published November 17, 2014.
  Fact 0.18: The manga Shigatsu wa Kimi no Uso: Coda is written by Naoshi Arakawa.
  Fact 0.19: The manga Shigatsu wa Kimi no Uso: Coda has the Shōnen demographic.
  Fact 0.20: The manga Shigatsu wa Kimi no Uso: Coda's original run was February 25, 2015 to October 28, 2015.
  Fact 0.21: The original video animation Moments is directed by Kazuya Iwata.
  Fact 0.22: The original video animation Moments is written by Takao Yoshioka.
  Fact 0.23: The original video animation Moments has music by Masaru Yokoyama.
  Fact 0.24: The original video animation Moments is by studio A-1 Pictures.
  Fact 0.25: The original video animation Moments was released May 15, 2015.
  Fact 0.26: The original video animation Moments has a runtime of 23 minutes.
  Fact 0.27: There is a live-action film titled Your Lie in April.

Retrieved items:
<item 1>
Your Lie in April
The letter states that she was in love with Kōsei the whole time , and that her supposed feelings towards Watari were fabricated in order to get closer to Kōsei
without emotionally injuring Tsubaki , who also has limerent feelings towards Kōsei . Although at first consumed with grief , Kōsei finds comfort in Tsubaki ,
who promises not to abandon him . Characters ( edit ) Kōsei Arima ( 有馬 公 生 , Arima Kōsei ) Voiced by : Natsuki Hanae ( Japanese ) ; Max Mittelman ( English ) &
Piano performance by : Tomoki Sakata Played by : Kento Yamazaki ( live - action film ) ; Shintarō Anzai ( stage play ) Kōsei is a former child prodigy in
playing piano , dubbed the `` Human Metronome '' for his near - inhuman mechanical accuracy , a product of his mother 's overly strict methods of teaching . His
ability to play the piano with unparalleled precision led him to win many competitions across Japan and even be invited to play abroad .
</item 1>

<item 2>
Your Lie in April
Your Lie in April - Wikipedia Your Lie in April Jump to : navigation , search This article is about the manga and anime . For the live - action film adaptation
, see Your Lie in April ( film ) . Your Lie in April Cover of the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono 四 月 は 君 の 嘘 ( Shigatsu wa Kimi
no Uso ) Manga Written by Naoshi Arakawa Published by Kodansha English publisher Kodansha Comics Demographic Shōnen Magazine Monthly Shōnen Magazine Original
run April 6 , 2011 -- February 6 , 2015 Volumes 11 ( List of volumes ) Anime television series Directed by Kyōhei Ishiguro Written by Takao Yoshioka Music by
Masaru Yokoyama Studio A-1 Pictures Licensed by Madman Entertainment Aniplex of America Anime Limited Original network Fuji TV ( Noitamina ) Original run
October 9 , 2014 -- March 19 ,
</item 2>

<item 3>
Your Lie in April
It premiered on October 10 , 2014 in Japan and ended on March 20 , 2015 . The first opening theme song is `` Hikaru Nara '' ( 光る なら , lit . If You Will Shine )
by Goose house and the first ending theme song is `` Kirameki '' ( キラメキ , lit . Sparkle ) by wacci . The second opening song is `` Nanairo Symphony '' ( 七色
シンフォニー , Nanairo Shinfonī , lit . `` Seven Colored Symphony '' ) by Coalamode and the second ending theme is `` Orange '' ( オレンジ , Orenji ) by 7 ! ! ( Seven
Oops ) . The original soundtracks are composed by Masaru Yokoyama . In North America , Aniplex of America licensed the series within the region , who
simulcasted the series on Crunchyroll , Aniplex Channel , Hulu and Viewster . In Australia and New Zealand , the series is licensed by Madman Entertainment ,
who streamed the series on AnimeLab .
</item 3>

<item 4>
Your Lie in April
. KODANSHA COMICS . Retrieved February 23 , 2015 . Jump up ^ `` Vertical to Release Your Lie in April Light Novel in July '' . Anime News Network . May 22 ,
2017 . Retrieved May 23 , 2017 . Jump up ^ `` Shōsetsu Shigatsu wa Kimi no Uso - 6 - nin no Etude '' 小説 四 月 は 君 の 嘘 6 人 の エチュード . Kodansha . Retrieved May 23 ,
2017 . Jump up ^ `` Shigatsu wa Kimi no Uso Manga Gets Noitamina Anime '' . Anime News Network . March 21 , 2014 . Retrieved March 21 , 2014 . ^ Jump up to : ``
Live - Action Your Lie in April Film Opens in September '' . Anime News Network . Retrieved 29 January 2016 .
</item 4>

<item 5>
Your Lie in April
2014 -- March 19 , 2015 Episodes 22 ( List of episodes ) Light novel Your Lie in April - A Six Person Etude Written by Yui Tokiumi Published by Kodansha English
publisher Vertical Published November 17 , 2014 Volumes Manga Shigatsu wa Kimi no Uso : Coda Written by Naoshi Arakawa Demographic Shōnen Original run February
25 , 2015 -- October 28 , 2015 Volumes Original video animation Moments Directed by Kazuya Iwata Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1
Pictures Released May 15 , 2015 Runtime 23 minutes Live - action film Your Lie in April ( film ) Anime and Manga portal Your Lie in April , known in Japan as
Shigatsu wa Kimi no Uso ( 四 月 は 君 の 嘘 ) or just simply Kimiuso , is a Japanese manga series written and illustrated by Naoshi Arakawa . The series was
serialized in Kodansha 's Monthly Shōnen Magazine from April 2011 to May 2015 .
</item 5>

<item 6>
Your Lie in April
Kaori Miyazono ( 宮園 かをり , Miyazono Kawori ) Voiced by : Risa Taneda ( Japanese ) ; Erica Lindbeck ( English ) & Violin performance by : Yūna Shinohara Played by
: Suzu Hirose ( live - action film ) ; Arisa Matsunaga ( stage play ) Kaori is Tsubaki 's classmate ; she is a free - spirited violinist who has drawn much
criticism from judging panels due to her unwillingness to adhere strictly to the score , but is highly favored by audiences that hear her playing . Kaori first
met Kōsei when she asked Tsubaki to set her up with Watari . As their friendship grew , she eventually convinced Kōsei to play the piano again , first as her
accompanist and later in a piano competition . It is revealed at the end of the anime , in a letter addressed to Kōsei , that she had asked to be set up with
Watari in order to meet Kōsei , knowing that Watari and Tsubaki were good friends of his .
</item 6>

<item 7>
Your Lie in April
Media ( edit ) Manga ( edit ) The Your Lie in April manga series is written and illustrated by Naoshi Arakawa , and began serialization on April 6 , 2011 in
Kodansha 's Monthly Shōnen Magazine . The first tankōbon was published by Kodansha on September 16 , 2011 , and the final volume was released in two editions ,
a regular edition and a limited edition , with the limited edition bundled with an OVA , and was released on May 15 , 2015 . The series is licensed in English
in North America by Kodansha Comics , and the first volume was published on April 21 , 2015 . A spin - off manga titled Shigatsu wa Kimi no Uso : Coda was
bundled with the release of the Blu - ray release of the anime series . It was later published in tankōbon format on August 17 , 2016 . Volume list ( edit ) No
. Japanese release date Japanese ISBN English release date English ISBN September 16 , 2011 ISBN 978 - 4 - 06 - 371301 - 5 April 21 ,
</item 7>

<item 8>
Your Lie in April
An anime television series adaptation by A-1 Pictures aired from October 2014 to March 2015 on Fuji TV 's Noitamina block . A live - action film adaptation of
the same name was released in September 2016 . Contents ( hide ) 1 Plot 2 Characters 3 Media 3.1 Manga 3.1. 1 Volume list 3.2 Anime 3.3 Live - action film 3.4
Stage play 4 Reception 5 References 6 External links Plot ( edit ) Piano prodigy Kōsei Arima dominated various music competitions and has become famous among
child musicians , but also controversial . After the death of his abusive mother , who was also his coldhearted instructor who forced him to play the piano
emotionlessly , he had a mental breakdown at the age of eleven while performing at a piano recital which resulted in him no longer being able to hear the sound
of his piano even though his hearing was perfectly fine .
</item 8>

<item 9>
Your Lie in April
When his mother died , the resulting psychological trauma caused him to be unable to hear the sound of his piano playing , and he gave up on it . Two years
later , he takes up the piano again after being convinced by Kaori Miyazono to become her accompanist . Influenced by her emotional and unrestrained playing
style , Kōsei eventually finds himself falling in love with her . He , however , does not confess his feelings due to her liking his best friend , Ryota Watari
. It is later revealed in Shigatsu wa Kimi no Uso : Coda , a side story from Kōsei and Tsubaki 's childhood , that his inspiration for playing so beautifully in
his first competition was to cheer up Tsubaki after the passing of her grandmother .
</item 9>

<item 10>
Your Lie in April
Kodansha . Retrieved January 27 , 2016 . Jump up ^ `` Your Lie in April 9 '' . Penguin Random House . Retrieved April 11 , 2016 . Jump up ^ 四 月 は 君 の 嘘 ( 10 ) (
Your Lie in April ( 10 ) ) ( in Japanese ) . Kodansha . Retrieved January 27 , 2016 . Jump up ^ `` Your Lie in April 10 '' . Penguin Random House . Retrieved
November 6 , 2016 . Jump up ^ `` Your Lie in April 11 '' . Penguin Random House . Retrieved April 11 , 2016 . Jump up ^ `` Goose House to Perform Opening ,
Cover Songs for Shigatsu wa Kimi no Uso Anime '' . Anime News Network . August 15 , 2014 . Retrieved August 3 , 2017 . Jump up ^ `` Coala Mode , 7 ! ! Perform
Theme Songs For Your Lie in April Anime 's 2nd Half '' . Anime News Network .
</item 10>
=== END ITEM id=ceb18ff6eb57f363 ===

=== ITEM id=6a0860b82af6b18e ===
Gold sentences and facts:
Sentence 0: The Cylinder 's text has traditionally been seen by biblical scholars as corroborative evidence of Cyrus ' policy of the repatriation of the Jewish people following their Babylonian captivity ( an act that the Book of Ezra attributes to Cyrus ) , as the text refers to the restoration of cult sanctuaries and repatriation of deported peoples . This interpretation has been disputed , as the text identifies only Mesopotamian sanctuaries , and makes no mention of Jews , Jerusalem , or Judea . The Cylinder has also been referred to by Mohammad Reza Pahlavi , the last Shah of Iran as the first declaration of universal human rights , a view rejected by some historians as anachronistic and a misunderstanding of the Cylinder 's generic nature as a typical statement made by a new monarch at the beginning of his reign . Neil MacGregor , Director of the British Museum , has stated that the cylinder was `` the first attempt we know about running a society , a state with different nationalities and faiths -- a new kind of statecraft . '' It was adopted as a national symbol of Iran by the Imperial State which put it on display in Tehran in 1971 to commemorate 2,500 year celebration of the Persian Empire . On October 14 , the Mohammad Reza Shah 's sister , Princess Ashraf Pahlavi , presented the United Nations Secretary General U Thant with a replica of the Cylinder . The princess asserted that `` the heritage of Cyrus was the heritage of human understanding , tolerance , courage , compassion and , above all , human liberty '' .
  Fact 0.0: The Cylinder's text has traditionally been seen by biblical scholars as corroborative evidence of Cyrus' policy of repatriating the Jewish people after their Babylonian captivity.
  Fact 0.1: The Book of Ezra attributes the repatriation of the Jewish people after their Babylonian captivity to Cyrus.
  Fact 0.2: The Cylinder's text refers to the restoration of cult sanctuaries.
  Fact 0.3: The Cylinder's text refers to the repatriation of deported peoples.
  Fact 0.4: The interpretation of the Cylinder's text as evidence of Cyrus' repatriation policy has been disputed.
  Fact 0.5: The Cylinder's text identifies only Mesopotamian sanctuaries.
  Fact 0.6: The Cylinder's text makes no mention of Jews, Jerusalem, or Judea.
  Fact 0.7: Mohammad Reza Pahlavi, the last Shah of Iran, referred to the Cylinder as the first declaration of universal human rights.
  Fact 0.8: Some historians reject the view that the Cylinder is the first declaration of universal human rights.
  Fact 0.9: Some historians consider the view that the Cylinder is the first declaration of universal human rights anachronistic.
  Fact 0.10: Some historians consider the view that the Cylinder is the first declaration of universal human rights a misunderstanding of the Cylinder's generic nature.
  Fact 0.11: The Cylinder is a typical statement made by a new monarch at the beginning of his reign.
  Fact 0.12: Neil MacGregor, Director of the British Museum, has stated that the Cylinder was "the first attempt we know about running a society, a state with different nationalities and faiths -- a new kind of statecraft."
  Fact 0.13: The Imperial State of Iran adopted the Cylinder as a national symbol of Iran.
  Fact 0.14: The Imperial State put the Cylinder on display in Tehran in 1971.
  Fact 0.15: The 1971 display in Tehran commemorated the 2,500 year celebration of the Persian Empire.
  Fact 0.16: On October 14, Princess Ashraf Pahlavi, the sister of Mohammad Reza Shah, presented United Nations Secretary General U Thant with a replica of the Cylinder.
  Fact 0.17: Princess Ashraf Pahlavi asserted that "the heritage of Cyrus was the heritage of human understanding, tolerance, courage, compassion and, above all, human liberty."

Retrieved items:
<item 1>
United States Bill of Rights
Elbridge Gerry wrote the most popular Anti-Federalist tract , `` Hon . Mr. Gerry 's Objections '' , which went through 46 printings ; the essay particularly
focused on the lack of a bill of rights in the proposed constitution . Many were concerned that a strong national government was a threat to individual rights
and that the president would become a king . Jefferson wrote to Madison advocating a Bill of Rights : `` Half a loaf is better than no bread . If we can not
secure all our rights , let us secure what we can . '' The pseudonymous Anti-Federalist `` Brutus '' wrote , We find they have , in the ninth section of the
first article declared , that the writ of habeas corpus shall not be suspended , unless in cases of rebellion -- that no bill of attainder , or ex post facto
law , shall be passed -- that no title of nobility shall be granted by the United States , etc . If every thing which is not given is reserved , what propriety
is there in these exceptions ?
</item 1>

<item 2>
United States Bill of Rights
1776 Virginia Declaration of Rights `` All men are created equal '' Life , Liberty and the pursuit of Happiness Freedom of the press Freedom of religion Consent
of the governed Baseless search and seizure Cruel and unusual punishment Speedy trial 1776 Virginia Constitution 1785 Mount Vernon Conference 1787
Constitutional Convention Virginia Ratifying Convention Co-father , United States Bill of Rights history Founding Father Writings inspired 1789 Declaration of
the Rights of Man and of the Citizen ( France ) 1789 United States Bill of Rights Life Chopawamsic plantation Gunston Hall On slavery Ohio Company Legacy George
Mason Memorial George Mason University George Mason Stadium George Mason Memorial Bridge George Mason High School 18 - cent postage stamp Related Age of
Enlightenment American Enlightenment American Revolution patriots Wilson v. Mason Hollin Hall Woodbridge plantation Mason 's Island Family George Mason V ( son
) William Mason ( son ) Thomson Mason ( son ) John Mason ( son ) Thomas Mason ( son ) George Mason III ( father ) Thomson Mason ( brother ) George Mason II (
grandfather ) VIAF : 186307101 LCCN : n83073936 GND : 4187279 - 4 Retrieved from `` https://en.wikipedia.
</item 2>

<item 3>
United States Bill of Rights
Bordewich , Fergus M. The First Congress : How James Madison , George Washington , and a Group of Extraordinary Men Invented the Government ( 2016 ) on 1789 --
1791 . Cogan , Neil H. ( 2015 ) . The Complete Bill of Rights : The Drafts , Debates , Sources , and Origins . Second edition . New York : Oxford University
Press . Schwartz , Bernard ( 1992 ) . The great rights of mankind : a history of the American Bill of Rights . Rowman & Littlefield . ISBN 978 - 0 - 945612 - 28
- 5 -- via Google Books . Smith , Rich ( 2007 ) . The Bill of Rights : Defining Our Freedoms . ABDO . ISBN 978 - 1 - 59928 - 913 - 7 -- via Google Books . Stair
, Nancy L. ( 2003 ) . The Bill of Rights : a primary source investigation into the first ten amendments of the Constitution . The Rosen Publishing Group . p. 53
.
</item 3>

<item 4>
United States Bill of Rights
`` American Indians and the Constitution '' . Encyclopedia of the American Constitution . -- via HighBeam Research ( subscription required ) . Retrieved July 16
, 2013 . Jump up ^ `` Primary Documents in American History '' , Library of Congress ^ Jump up to : Jeffrey Jowell and Jonathan Cooper ( 2002 ) . Understanding
Human Rights Principles . Oxford and Portland , Oregon : Hart Publishing . p. 180 . ISBN 9781847313157 . Retrieved 16 March 2017 . ^ Jump up to : Loveland , Ian
( 2002 ) . `` Chapter 18 -- Human Rights I : Traditional Perspectives '' . Constitutional Law , Administrative Law , and Human Rights : A Critical Introduction
( Seventh ed . ) . London : Oxford University Press . p. 559 . ISBN 9780198709039 . Retrieved 16 March 2017 . ^ Jump up to : Jayawickrama , Nihal ( 2002 ) . The
Judicial Application of Human Rights Law : National , Regional and International Jurisprudence . Cambridge : Cambridge University Press . p.
</item 4>

<item 5>
United States Bill of Rights
Constitution , and crafted to address the objections raised by Anti-Federalists , the Bill of Rights amendments add to the Constitution specific guarantees of
personal freedoms and rights , clear limitations on the government 's power in judicial and other proceedings , and explicit declarations that all powers not
specifically delegated to Congress by the Constitution are reserved for the states or the people . The concepts codified in these amendments are built upon
those found in several earlier documents , including the Virginia Declaration of Rights and the English Bill of Rights , along with earlier documents such as
Magna Carta ( 1215 ) . In practice , the amendments had little impact on judgments by the courts for the first 150 years after ratification . On June 8 , 1789 ,
Representative James Madison introduced nine amendments to the Constitution in the House of Representatives . Among his recommendations Madison proposed opening
up the Constitution and inserting specific rights limiting the power of Congress in Article One , Section 9 . Seven of these limitations would become part of
the ten ratified Bill of Rights amendments .
</item 5>

<item 6>
Fundamental rights in India
A notable development during that period having significant effect on the Indian constitution took place on 10 December 1948 when the United Nations General
Assembly adopted the Universal Declaration of Human Rights and called upon all member states to adopt these rights in their respective constitutions . The
fundamental rights were included in the First Draft Constitution ( February 1948 ) , the Second Draft Constitution ( 17 October 1948 ) and final Third Draft
Constitution ( 26 November 1949 ) , prepared by the Drafting Committee . Significance and characteristics ( edit ) The fundamental rights were included in the
constitution because they were considered essential for the development of the personality of every individual and to preserve human dignity . The writers of
the constitution regarded democracy of no avail if civil liberties , like freedom of speech and religion were not recognised and protected by the State .
According to them , `` democracy '' is , in essence , a government by opinion and therefore , the means of formulating public opinion should be secured to the
people of a democratic nation . For this purpose , the constitution guaranteed to all the citizens of India the freedom of speech and expression and various
other freedoms in the form of the fundamental rights .
</item 6>

<item 7>
United States Bill of Rights
In response , Hamilton argued that the Constitution was inherently different : Bills of rights are in their origin , stipulations between kings and their
subjects , abridgments of prerogative in favor of privilege , reservations of rights not surrendered to the prince . Such was the Magna Charta , obtained by the
Barons , swords in hand , from King John . Massachusetts compromise Main article : Massachusetts Compromise George Washington 's 1788 letter to the Marquis de
Lafayette observed , `` the Convention of Massachusetts adopted the Constitution in toto ; but recommended a number of specific alterations and quieting
explanations . '' Source : Library of Congress In December 1787 and January 1788 , five states -- Delaware , Pennsylvania , New Jersey , Georgia , and
Connecticut -- ratified the Constitution with relative ease , though the bitter minority report of the Pennsylvania opposition was widely circulated . In
contrast to its predecessors , the Massachusetts convention was angry and contentious , at one point erupting into a fistfight between Federalist delegate
Francis Dana and Anti-Federalist Elbridge Gerry when the latter was not allowed to speak .
</item 7>

<item 8>
United States Bill of Rights
Historians continue to debate the degree to which Madison considered the amendments of the Bill of Rights necessary , and to what degree he considered them
politically expedient ; in the outline of his address , he wrote , `` Bill of Rights -- useful -- not essential -- '' . On the occasion of his April 30 , 1789
inauguration as the nation 's first president , George Washington addressed the subject of amending the Constitution . He urged the legislators , whilst you
carefully avoid every alteration which might endanger the benefits of an united and effective government , or which ought to await the future lessons of
experience ; a reverence for the characteristic rights of freemen , and a regard for public harmony , will sufficiently influence your deliberations on the
question , how far the former can be impregnably fortified or the latter be safely and advantageously promoted . Crafting amendments James Madison introduced a
series of Constitutional amendments in the House of Representatives for consideration . Among his proposals was one that would have added introductory language
stressing natural rights to the preamble . Another would apply parts of the Bill of Rights to the states as well as the federal government . Several sought to
protect individual personal rights by limiting various Constitutional powers of Congress .
</item 8>

<item 9>
United States Bill of Rights
Several sought to protect individual personal rights by limiting various Constitutional powers of Congress . Like Washington , Madison urged Congress to keep
the revision to the Constitution `` a moderate one '' , limited to protecting individual rights . Madison was deeply read in the history of government and used
a range of sources in composing the amendments . The English Magna Carta of 1215 inspired the right to petition and to trial by jury , for example , while the
English Bill of Rights of 1689 provided an early precedent for the right to keep and bear arms ( although this applied only to Protestants ) and prohibited
cruel and unusual punishment . The greatest influence on Madison 's text , however , was existing state constitutions . Many of his amendments , including his
proposed new preamble , were based on the Virginia Declaration of Rights drafted by Anti-Federalist George Mason in 1776 . To reduce future opposition to
ratification , Madison also looked for recommendations shared by many states . He did provide one , however , that no state had requested : `` No state shall
violate the equal rights of conscience , or the freedom of the press , or the trial by jury in criminal cases . ''
</item 9>

<item 10>
Graph query row:
author: George Mason
declaration: Virginia Declaration of Rights
connection: RELATED_TO
related: Patriots
</item 10>
=== END ITEM id=6a0860b82af6b18e ===
