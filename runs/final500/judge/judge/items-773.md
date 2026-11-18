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

=== ITEM id=ae6ce4a0e8d37f70 ===
Gold sentences and facts:
Sentence 0: A board of directors is a recognized group of people who jointly oversee the activities of an organization , which can be either a for - profit business , nonprofit organization , or a government agency . Such a board 's powers , duties , and responsibilities are determined by government regulations ( including the jurisdiction 's corporations law ) and the organization 's own constitution and bylaws . These authorities may specify the number of members of the board , how they are to be chosen , and how often they are to meet .
  Fact 0.0: A board of directors is a recognized group of people.
  Fact 0.1: A board of directors jointly oversees the activities of an organization.
  Fact 0.2: An organization overseen by a board of directors can be a for-profit business, a nonprofit organization, or a government agency.
  Fact 0.3: A board's powers, duties, and responsibilities are determined by government regulations, including the jurisdiction's corporations law.
  Fact 0.4: A board's powers, duties, and responsibilities are determined by the organization's own constitution and bylaws.
  Fact 0.5: The authorities over a board may specify the number of members of the board.
  Fact 0.6: The authorities over a board may specify how board members are chosen.
  Fact 0.7: The authorities over a board may specify how often the board is to meet.

Retrieved items:
<item 1>
Board of directors
^ Jump up to : Robert 2011 , p. 481 - 483 . Jump up ^ McNamara , Carter . `` Overview of Roles and Responsibilities of Corporate Board of Directors '' . Free
Management Library . Authenticity Consulting , LLC . Retrieved 2008 - 01 - 26 . Jump up ^ `` Basic Role of the Board '' . Governance Basics . Institute on
Governance ( Canada ) . Archived from the original on 30 December 2007 . Retrieved 2008 - 01 - 27 . ^ Jump up to : Robert 2011 , p. 484 . Jump up ^ This section
was developed from numerous definitions in USLegal.com , BusinessDictionary.com , Dictionary.com , The Free Dictionary by Farlex ( `` inside director '' ; ``
executive director '' ; `` outside director '' ; `` nonexecutive director '' ) , Macmillan Dictionary , and Economics-dictionary.com . Jump up ^ `` Executive
Director '' . Investopedia . Retrieved 24 May 2013 . Jump up ^ `` Outside Director '' . Investopedia . Retrieved 24 May 2013 .
</item 1>

<item 2>
Board of directors
In most common law countries , the powers of the board are vested in the board as a whole , and not in the individual directors . However , in instances an
individual director may still bind the company by his acts by virtue of his ostensible authority ( see also : the rule in Turquand 's Case ) . Duties ( edit )
Main articles : Directors ' duties and Fiduciary duties Because directors exercise control and management over the organization , but organizations are ( in
theory ) run for the benefit of the shareholders , the law imposes strict duties on directors in relation to the exercise of their duties . The duties imposed
on directors are fiduciary duties , similar to those that the law imposes on those in similar positions of trust : agents and trustees . The duties apply to
each director separately , while the powers apply to the board jointly . Also , the duties are owed to the company itself , and not to any other entity . This
does not mean that directors can never stand in a fiduciary relationship to the individual shareholders ; they may well have such a duty in certain
circumstances . `` Proper purpose '' ( edit ) Directors must exercise their powers for a proper purpose .
</item 2>

<item 3>
Board of directors
Typical inside directors are : A chief executive officer ( CEO ) who may also be chairman of the board Other executives of the organization , such as its chief
financial officer ( CFO ) or executive vice president Large shareholders ( who may or may not also be employees or officers ) Representatives of other
stakeholders such as labor unions , major lenders , or members of the community in which the organization is located An inside director who is employed as a
manager or executive of the organization is sometimes referred to as an executive director ( not to be confused with the title executive director sometimes used
for the CEO position in some organizations ) . Executive directors often have a specified area of responsibility in the organization , such as finance ,
marketing , human resources , or production . Outside director ( edit ) Main article : Independent director An outside director is a member of the board who is
not otherwise employed by or engaged with the organization , and does not represent any of its stakeholders . A typical example is a director who is president
of a firm in a different industry . Outside directors are not employees of the company or affiliated with it in any other way . Outside directors bring outside
experience and perspectives to the board .
</item 3>

<item 4>
Board of directors
crisis Sales Security Service Strategic Supply chain Systems administrator Talent Technology Organization ( show ) Architecture Behavior Communication Culture
Conflict Development Engineering Hierarchy Patterns Space Structure Trade ( show ) Business analysis Business ethics Business plan Business judgment rule
Consumer behaviour Business operations International business Business model International trade Business process Business statistics Business and economics
portal Typical duties of boards of directors include : governing the organization by establishing broad policies and setting out strategic objectives ;
selecting , appointing , supporting and reviewing the performance of the chief executive ( of which the titles vary from organization to organization ; the
chief executive may be titled chief executive officer , president or executive director ) ; terminating the chief executive ; ensuring the availability of
adequate financial resources ; approving annual budgets ; accounting to the stakeholders for the organization 's performance ; setting the salaries ,
compensation and benefits of senior management ; The legal responsibilities of boards and board members vary with the nature of the organization , and between
jurisdictions . For companies with publicly trading stock , these responsibilities are typically much more rigorous and complex than for those of other types .
Typically , the board chooses one of its members to be the chairman ( more usually now called the `` chair '' or `` chairperson '' ) , who holds whatever title
is specified in the bylaws or articles of association .
</item 4>

<item 5>
Board of directors
According to Investopedia , some analysts think the ideal size is seven . State law may specify a minimum number of directors , maximum number of directors ,
and qualifications for directors ( e.g. whether board members must be individuals or may be business entities ) . Committees ( edit ) While a board may have
several committees , two -- the compensation committee and audit committee -- are critical and must be made up of at least three independent directors and no
inside directors . Other common committees in boards are nominating and governance . Compensation ( edit ) Directors of Fortune 500 companies received median
pay of $234,000 in 2011 . Directorship is a part - time job . A recent National Association of Corporate Directors study found directors averaging just 4.3
hours a week on board work . Surveys indicate that about 20 % of nonprofit foundations pay their board members , and 2 % of American nonprofit organizations do
. 80 % of nonprofit organizations require board members to personally contribute to the organization , as BoardSource recommends . This percentage has increased
in recent years .
</item 5>

<item 6>
Board of directors
Inside director -- a director who , in addition to serving on the board , has a meaningful connection to the organization Outside director -- a director who ,
other than serving on the board , has no meaningful connections to the organization Executive director -- an inside director who is also an executive with the
organization . The term is also used , in a completely different sense , to refer to a CEO Non-executive director -- an inside director who is not an executive
with the organization Shadow or de facto director -- an individual who is not a named director but who nevertheless directs or controls the organization Nominee
director -- an individual who is appointed by a shareholder , creditor or interest group ( whether contractually or by resolution at a company meeting ) and who
has a continuing loyalty to the appointor / s or other interest in the appointing company Individual directors often serve on more than one board . This
practice results in an interlocking directorate , where a relatively small number of individuals have significant influence over a large number of important
entities . This situation can have important corporate , social , economic , and legal consequences , and has been the subject of significant research .
</item 6>

<item 7>
Board of directors
( eds ) , Comparative Corporate Governance : The State of the Art and Emerging Research ( Clarendon 1998 ) KJ Hopt and PC Leyens , ' Board Models in Europe --
Recent Developments of Internal Corporate Governance Structures in Germany , the United Kingdom , France , and Italy ' ( 2004 ) EGCI Working Paper Robert ,
Henry M. ; et al. ( 2011 ) . Robert 's Rules of Order Newly Revised ( 11th ed . ) . Philadelphia , PA : Da Capo Press . ISBN 978 - 0 - 306 - 82020 - 5 .
Archived from the original on 26 July 2017 . External links ( edit ) Web site of the Board of a large U.S. university , illustrating a typical board 's
composition , duties , concerns , etc . Guidance on director 's duties ( Lemon & Co ) National Association of Corporate Directors European Directors and Board
Members Association Corporate Governance Board Leadership Training , Global Corporate Governance Forum BoardofDirectors.com.
</item 7>

<item 8>
Board of directors
The setup of a board of directors vary widely across organizations and may include provisions that are applicable to corporations , in which the `` shareholders
'' are the members of the organization . A difference may be that the membership elects the officers of the organization , such as the president and the
secretary , and the officers become members of the board in addition to the directors and retain those duties on the board . The directors may also be
classified as officers in this situation . There may also be ex-officio members of the board , or persons who are members due to another position that they hold
. These ex-officio members have all the same rights as the other board members . Members of the board may be removed before their term is complete . Details on
how they can be removed are usually provided in the bylaws . If the bylaws do not contain such details , the section on disciplinary procedures in Robert 's
Rules of Order may be used . Corporations ( edit ) In a publicly held company , directors are elected to represent and are legally obligated as fiduciaries to
represent owners of the company -- the shareholders / stockholders .
</item 8>

<item 9>
Board of directors
Process and structure ( edit ) The process for running a board , sometimes called the board process , includes the selection of board members , the setting of
clear board objectives , the dissemination of documents or board package to the board members , the collaborative creation of an agenda for the meeting , the
creation and follow - up of assigned action items , and the assessment of the board process through standardized assessments of board members , owners , and
CEOs . The science of this process has been slow to develop due to the secretive nature of the way most companies run their boards , however some
standardization is beginning to develop . Some who are pushing for this standardization in the USA are the National Association of Corporate Directors ,
McKinsey Consulting and The Board Group . Board meetings ( edit ) A board of directors conducts its meetings according to the rules and procedures contained in
its governing documents . These procedures may allow the board to conduct its business by conference call or other electronic means . They may also specify how
a quorum is to be determined . Most organizations have adopted Robert 's Rules of Order as its guide to supplement its own rules .
</item 9>

<item 10>
Board of directors
However , in membership organizations , the members elect the president of the organization and the president becomes the chair of the board , unless the bylaws
say otherwise . Directors ( edit ) The directors of an organization are the persons who are members of its board . Several specific terms categorize directors
by the presence or absence of their other relationships to the organization . Inside director ( edit ) An inside director is a director who is also an employee
, officer , chief executive , major shareholder , or someone similarly connected to the organization . Inside directors represent the interests of the entity 's
stakeholders , and often have special knowledge of its inner workings , its financial or market position , and so on .
</item 10>
=== END ITEM id=ae6ce4a0e8d37f70 ===

=== ITEM id=95bf24a6d1499399 ===
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
Sahara
Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba Mountains Nubia The
Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf of Guinea Niger Basin
Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern Highlands Rhodesia North South
Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay Hydra Bay Macro-regions Aethiopia
Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater Middle East Islands of Africa
List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland Northeast Africa Portuguese - speaking
African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa Coordinates : 23 ° N 13 ° E ﻿ / ﻿ 23 ° N
13 ° E ﻿ / 23 ;
</item 1>

<item 2>
Petroleum industry in Nigeria
An open - air market for illegal crude oil operates off the Niger Delta , called the Togo Triangle . Natural gas ( edit ) Natural gas reserves are well over
5,300 km ( 187 × 10 ^ cu ft ) , the gas reserves are three times as substantial as the crude oil reserves . The biggest natural gas initiative is the Nigerian
Liquified Natural Gas Company , which is operated jointly by several companies and the state . It began exploration and production in 1999 . Chevron is also
attempting to create the Escravos Gas Utilization project which will be capable of producing 4,500,000 m ( 160 × 10 ^ cu ft ) per day . gas reserves . In 2008 ,
the government prepared a Gas Master Plan that was intended to promote natural gas production and encourage the supply of natural gas to domestic power stations
so as to help alleviate the country 's electricity shortages . There is also an export gas pipeline , known as the West African Gas Pipeline , in the works but
has encountered numerous setbacks . The pipeline would allow for transportation of natural gas to Benin , Ghana , Togo , and Cote d'Ivoire .
</item 2>

<item 3>
Petroleum industry in Nigeria
The most productive region of the nation is the coastal Niger Delta Basin in the Niger Delta or `` South - south '' region which encompasses 78 of the 159 oil
fields . Most of Nigeria 's oil fields are small and scattered , and as of 1990 , these small unproductive fields accounted for 62.1 % of all Nigerian
production . This contrasts with the sixteen largest fields which produced 37.9 % of Nigeria 's petroleum at that time . As a result of the numerous small
fields , an extensive and well - developed pipeline network has been engineered to transport the crude oil . Also because of the lack of highly productive
fields , money from the jointly operated ( with the federal government ) companies is constantly directed towards petroleum exploration and production . Nigeria
's petroleum is classified mostly as `` light '' and `` sweet '' , as the oil is largely free of sulphur . Nigeria is the largest producer of sweet oil in OPEC
. This sweet oil is similar in composition to petroleum extracted from the North Sea . This crude oil is known as `` Bonny light '' .
</item 3>

<item 4>
Dead Sea
Congolese rainforests Ouaddaï highlands Ennedi Plateau East Africa African Great Lakes Albertine Rift East African Rift Great Rift Valley Gregory Rift Rift
Valley lakes Swahili coast Virunga Mountains Zanj Horn of Africa Afar Triangle Al - Habash Barbara Danakil Alps Danakil Desert Ethiopian Highlands Gulf of Aden
Gulf of Tadjoura Indian Ocean islands Comoros Islands North Africa Maghreb Barbary Coast Bashmur Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile
Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba Mountains Nubia The Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold
Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf of Guinea Niger Basin Guinean Forests of West Africa Niger Delta Inner Niger Delta
Southern Africa Madagascar Central Highlands ( Madagascar ) Northern Highlands Rhodesia North South Thembuland Succulent Karoo Nama Karoo Bushveld Highveld
Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay Hydra Bay Macro-regions Aethiopia Arab world Commonwealth realm East African montane
forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater Middle East
</item 4>

<item 5>
Sub-Saharan Africa
Major suppliers are South Africa , Ghana , Zimbabwe , Tanzania , Guinea , and Mali . South Africa had been first in the world in terms of gold production since
1905 , but in 2007 it moved to second place , according to GFMS , the precious metals consultancy . Uranium is major commodity from the region . Significant
suppliers are Niger , Namibia , and South Africa . Namibia was the number one supplier from Sub-Saharan Africa in 2008 . The region produces 49 % of the world
's diamonds . By 2015 , it is estimated that 25 % of North American oil will be from Sub-Saharan Africa , ahead of the Middle East . Sub-Saharan Africa has been
the focus of an intense race for oil by the West , China , India , and other emerging economies , even though it holds only 10 % of proven oil reserves , less
than the Middle East . This race has been referred to as the second Scramble for Africa . All reasons for this global scramble come from the reserves ' economic
benefits . Transportation cost is low and no pipelines have to be laid as in Central Asia .
</item 5>

<item 6>
Petroleum industry in Nigeria
Mobil operates primarily out of the Qua Iboe Terminal in Akwa Ibom State , while Chevron owns the Escravos Terminal located in Delta State and has a storage
capacity of 3.6 million barrels ( 570,000 m ) . Agip operates the Brass Terminal in Brass , a town 113 kilometres ( 70 miles ) southwest of Port Harcourt and
has a storage capacity of 3,558,000 barrels ( 565,700 m ) . Texaco operates the Pennington Terminal . Offshore ( edit ) Oil companies in Africa investigate
offshore production as an alternative area of production . Deepwater production mainly involves underwater drilling that exists 400 metres ( 1,300 ft ) or more
below the surface of the water . By expanding to deep water drilling the possible sources for finding new oil reserves is expanded . Through the introduction of
deep water drilling 50 % more oil is extracted than before the new forms of retrieving the oil . Angola and Nigeria are the largest oil producers in Africa . In
Nigeria , the deepwater sector still has a large avenue to expand and develop .
</item 6>

<item 7>
Sub-Saharan Africa
Transportation cost is low and no pipelines have to be laid as in Central Asia . Almost all reserves are offshore , so political turmoil within the host country
will not directly interfere with operations . Sub-Saharan oil is viscous , with a very low sulfur content . This quickens the refining process and effectively
reduces costs . New sources of oil are being located in Sub-Saharan Africa more frequently than anywhere else . Of all new sources of oil , 1⁄3 are in
Sub-Saharan Africa . Agriculture ( edit ) Agricultural fields in Rwanda 's Eastern Province Sub-Saharan Africa has more variety of grains than anywhere in the
world . Between 13,000 and 11,000 BCE wild grains began to be collected as a source of food in the cataract region of the Nile , south of Egypt . The collecting
of wild grains as source of food spread to Syria , parts of Turkey , and Iran by the eleventh millennium BCE . By the tenth and ninth millennia southwest Asians
domesticated their wild grains , wheat , and barley after the notion of collecting wild grains spread from the Nile . Numerous crops have been domesticated in
the region and spread to other parts of the world .
</item 7>

<item 8>
Sub-Saharan Africa
Australia 31 Af : Sudan 466100 2007 est . Ecuador 33 Af : Eq. Guinea 368500 2007 est . Vietnam 38 Af : DR Congo 261000 2008 est . 39 Af : Gabon 243900 2007 est
. 40 Af : Sth Africa 199100 2007 est . 45 Af : Chad 156000 2008 est . Germany 53 Af : Cameroon 87400 2008 est . France 56 E : France 71400 2007 60 Af : Ivory
Coast 54400 2008 est . _ Af : Africa 10780400 2011 Russia Source : CIA.gov , World Facts Book > Oil exporters . As of 2009 , fifty percent of Africa is rural
with no access to electricity . Africa generates 47 GW of electricity , less than 0.6 % of the global market share . Many countries are affected by power
shortages . Because of rising prices in commodities such as coal and oil , thermal sources of energy are proving to be too expensive for power generation .
</item 8>

<item 9>
Sub-Saharan Africa
Barbary Coast Bashmur Ancient Libya Atlas Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba
Mountains Nubia The Sudans Upper Egypt Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf
of Guinea Niger Basin Guinean Forests of West Africa Niger Delta Inner Niger Delta Southern Africa Madagascar Central Highlands ( Madagascar ) Northern
Highlands Rhodesia North South Thembuland Succulent Karoo Nama Karoo Bushveld Highveld Fynbos Cape Floristic Region Kalahari Desert Okavango Delta False Bay
Hydra Bay Macro-regions Aethiopia Arab world Commonwealth realm East African montane forests Eastern Desert Equatorial Africa Françafrique Gibraltar Arc Greater
Middle East Islands of Africa List of countries where Arabic is an official language Mediterranean Basin MENA MENASA Middle East Mittelafrika Negroland
Northeast Africa Portuguese - speaking African countries Sahara Sahel Sub-Saharan Africa Sudan ( region ) Sudanian Savanna Tibesti Mountains Tropical Africa
Regions of Asia Central Greater Middle East Aral Sea Aralkum Desert Caspian Sea Dead Sea Sea of Galilee Transoxiana Turan Greater Khorasan
</item 9>

<item 10>
Sub-Saharan Africa
According to the World Bank , the economic growth rate in the region had risen to 4.7 % in 2013 , with a rate of 5.2 % forecasted for 2014 . This continued rise
was attributed to increasing investment in infrastructure and resources as well as steady expenditure per household . Energy and power ( edit ) Main article :
Mineral industry of Africa Oil production by country ( with other key actors of African or oil economy ) Rank Area bb / day Year Like ... _ W : World 85540000
2007 est . 01 E : Russia 9980000 2007 est . 02 Ar : Saudi Arb 9200000 2008 est . 04 As : Libya 4725000 2008 est . Iran 10 Af : Nigeria / Africa 2352000 2011 est
. Norway 15 Af : Algeria 2173000 2007 est . 16 Af : Angola 1910000 2008 est . 17 Af : Egypt 1845000 2007 est . 27 Af : Tunisia 664000 2007 est . Australia 31 Af
: Sudan 466100 2007 est .
</item 10>
=== END ITEM id=95bf24a6d1499399 ===

=== ITEM id=139996cd6d820173 ===
Gold sentences and facts:
Sentence 0: Country Name of animal Scientific name Pictures Ref . Algeria Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer ( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian manatee ( national aquatic animal ) Trichechus manatus Croatia Pine marten Martes martes Cuba Cuban trogon Priotelus temnurus Cyprus Cypriot mouflon Ovis orientalis Czech Republic Double - tailed lion Mythical Denmark Mute swan ( national bird ) Cygnus olor Small tortoiseshell ( national butterfly ) Aglais urticae Egypt Steppe eagle Aquila nipalensis Estonia Barn swallow ( national bird ) Hirundo rustica Eritrea Arabian camel Camelus dromedarius Ethiopia Lion Panthera leo Faroe Islands Eurasian oystercatcher ( national bird ) Haematopus ostralgus Finland Brown bear ( national animal ) Ursus arctos Whooper swan ( national bird ) Cygnus cygnus Holly blue ( national butterfly ) Celastrina argiolus European perch ( national fish ) Perca fluviatilis Seven - spot ladybird ( national insect ) Coccinella septempunctata France Gallic rooster Gallus gallus domesticus Germany Unspecified Eagle Members of Accipitridae Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator ) Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal ) Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG ) Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal ) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod ) Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle ( national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison
  Fact 0.0: In Algeria, the Fennec fox (Vulpes zerda) is listed as the animal.
  Fact 0.1: In Angola, the Red-crested turaco (Tauraco erythrolophus) is listed as the national bird.
  Fact 0.2: In Anguilla, the Zenaida dove (Zenaida aurita) is listed.
  Fact 0.3: In Antigua and Barbuda, the Fallow deer (Dama dama) is listed as the national animal.
  Fact 0.4: In Antigua and Barbuda, the Frigate (Fregata magnificens) is listed as the national bird.
  Fact 0.5: In Antigua and Barbuda, the Hawksbill turtle (Eretmochelys imbricata) is listed as the national sea creature.
  Fact 0.6: In Argentina, the Rufous hornero (Furnarius rufus) is listed.
  Fact 0.7: In Australia, the Red kangaroo (Macropus rufus) is listed as the national animal.
  Fact 0.8: In Australia, the Emu (Dromaius novaehollandiae) is listed as the national bird.
  Fact 0.9: In Austria, the Black eagle (Ictinaetus malaiensis) is listed.
  Fact 0.10: In Azerbaijan, the Karabakh horse (Equus ferus caballus) is listed.
  Fact 0.11: In Bangladesh, the Royal Bengal tiger (Panthera tigris tigris) is listed as the national animal.
  Fact 0.12: In Bangladesh, the Magpie robin (Copsychus saularis) is listed as the national bird.
  Fact 0.13: In Bangladesh, the Ilish (Tenualosa ilisha) is listed as the national fish.
  Fact 0.14: In Belarus, the European bison (Bison bonasus) is listed.
  Fact 0.15: In Belgium, the Lion (Panthera leo) is listed as the heraldic Leo Belgicus.
  Fact 0.16: In Belize, Baird's tapir (Tapirus bairdii) is listed as the national animal.
  Fact 0.17: In Belize, the Keel-billed toucan (Ramphastos sulfuratus) is listed as the national bird.
  Fact 0.18: In Bhutan, the Druk is listed as a mythical animal.
  Fact 0.19: In Bhutan, the Takin (Budorcas taxicolor) is listed.
  Fact 0.20: In Brazil, the Rufous-bellied thrush (Turdus rufiventris) is listed.
  Fact 0.21: In Cambodia, the Kouprey (Bos sauveli) is listed.
  Fact 0.22: In Canada, the North American beaver (Castor canadensis) is listed as the sovereignty animal symbol.
  Fact 0.23: In Canada, the Canadian horse (Equus ferus caballus) is listed as the national horse.
  Fact 0.24: In China, the Giant panda (Ailuropoda melanoleuca) is listed as the national animal.
  Fact 0.25: In China, the Chinese dragon is listed as a mythical national animal.
  Fact 0.26: In China, the Red-crowned crane (Grus japonensis) is listed as the national bird.
  Fact 0.27: In the Democratic Republic of the Congo, the Okapi (Okapia johnstoni) is listed.
  Fact 0.28: In Colombia, the Andean condor (Vultur gryphus) is listed.
  Fact 0.29: In Costa Rica, the Yigüirro (Turdus grayi) is listed as the national bird.
  Fact 0.30: In Costa Rica, the White-tailed deer (Odocoileus virginianus) is listed as the national animal.
  Fact 0.31: In Costa Rica, the West Indian manatee (Trichechus manatus) is listed as the national aquatic animal.
  Fact 0.32: In Croatia, the Pine marten (Martes martes) is listed.
  Fact 0.33: In Cuba, the Cuban trogon (Priotelus temnurus) is listed.
  Fact 0.34: In Cyprus, the Cypriot mouflon (Ovis orientalis) is listed.
  Fact 0.35: In the Czech Republic, the Double-tailed lion is listed as a mythical animal.
  Fact 0.36: In Denmark, the Mute swan (Cygnus olor) is listed as the national bird.
  Fact 0.37: In Denmark, the Small tortoiseshell (Aglais urticae) is listed as the national butterfly.
  Fact 0.38: In Egypt, the Steppe eagle (Aquila nipalensis) is listed.
  Fact 0.39: In Estonia, the Barn swallow (Hirundo rustica) is listed as the national bird.
  Fact 0.40: In Eritrea, the Arabian camel (Camelus dromedarius) is listed.
  Fact 0.41: In Ethiopia, the Lion (Panthera leo) is listed.
  Fact 0.42: In the Faroe Islands, the Eurasian oystercatcher (Haematopus ostralgus) is listed as the national bird.
  Fact 0.43: In Finland, the Brown bear (Ursus arctos) is listed as the national animal.
  Fact 0.44: In Finland, the Whooper swan (Cygnus cygnus) is listed as the national bird.
  Fact 0.45: In Finland, the Holly blue (Celastrina argiolus) is listed as the national butterfly.
  Fact 0.46: In Finland, the European perch (Perca fluviatilis) is listed as the national fish.
  Fact 0.47: In Finland, the Seven-spot ladybird (Coccinella septempunctata) is listed as the national insect.
  Fact 0.48: In France, the Gallic rooster (Gallus gallus domesticus) is listed.
  Fact 0.49: In Germany, the Unspecified Eagle (a member of Accipitridae) is listed.
  Fact 0.50: In Greece, the Phoenix is listed as a mythical animal.
  Fact 0.51: In Greece, the Dolphins (Delphinus) are listed.
  Fact 0.52: In Guatemala, the Quetzal (Pharomachrus mocinno) is listed.
  Fact 0.53: In Haiti, the Hispaniolan Trogan (Priotelus roseigaster) is listed.
  Fact 0.54: In Honduras, the White-tailed deer (Odocoileus virginianus) is listed.
  Fact 0.55: In Hungary, the Turul is listed as a mythical animal, most probably Falco cherrug.
  Fact 0.56: In Iceland, the Gyrfalcon (Falco rusticolus) is listed.
  Fact 0.57: In India, the Bengal tiger (Panthera tigris tigris) is listed as the National predator.
  Fact 0.58: In India, the Indian peafowl (Pavo cristatus) is listed as the National bird.
  Fact 0.59: In India, the King cobra (Ophiophagus hannah) is listed as the National reptile.
  Fact 0.60: In India, the Ganges river dolphin (Platanista gangetica gangetica) is listed as the National aquatic animal.
  Fact 0.61: In India, the Indian elephant (Elephas maximus indicus) is listed as the National heritage animal.
  Fact 0.62: In Indonesia, the Komodo dragon (Varanus komodoensis) is listed as the national animal.
  Fact 0.63: In Indonesia, the Garuda is listed as the historic national animal, a mythical one.
  Fact 0.64: In Iran, the Persian lion (Panthera leo persica) is listed.
  Fact 0.65: In Iraq, the Chukar partridge (Alectoris chukar) is listed.
  Fact 0.66: In Iraq, the Goat (Capra aegagrus) is listed for KRG.
  Fact 0.67: In Israel, the Hoopoe (Upupa epops) is listed as the national bird.
  Fact 0.68: In Jamaica, the Red-billed streamertail (Trochilus polytmus) is listed.
  Fact 0.69: In Japan, the Green pheasant (Phasianus versicolor) is listed as the national bird.
  Fact 0.70: In Japan, the Carp (Cyprinus carpio) is listed as the national fish.
  Fact 0.71: In Latvia, the White wagtail (Motacilla alba) is listed as the national bird.
  Fact 0.72: In Latvia, the Two-spotted ladybird (Adalia bipunctata) is listed as the national insect.
  Fact 0.73: In Lithuania, the White stork (Ciconia ciconia) is listed.
  Fact 0.74: In Macedonia, the Lion (Panthera leo) is listed as the animal in Macedonian heraldry.
  Fact 0.75: In Madagascar, the Zebu (Bos primigenius indicus) is listed as the animal in Malagasy heraldry.
  Fact 0.76: In Madagascar, the Ring-tailed lemur (Lemur catta) is listed as the national animal.
  Fact 0.77: In Malaysia, the Malayan tiger (Panthera tigris jacksoni) is listed as the national animal.
  Fact 0.78: In Malta, the Pharaoh Hound (Canis lupus familiaris) is listed as the national animal, with reference 1.
  Fact 0.79: In Moldova, the Aurochs (Bos primigenius) is listed as the national animal.
  Fact 0.80: In Mauritius, the Dodo (Raphus cucullatus) is listed.
  Fact 0.81: In Mexico, the Golden eagle (Aquila chrysaetos) is listed as the national animal.
  Fact 0.82: In Mexico, the Xoloitzcuintli (Canis lupus familiaris) is listed as the national dog.
  Fact 0.83: In Mexico, the Jaguar (Panthera onca) is listed as the national mammal.
  Fact 0.84: In Mexico, the Grasshopper (Sphenarium purpurascens) is listed as the national arthropod.
  Fact 0.85: In Mexico, the Vaquita (Phocoena sinus) is listed as the national marine mammal.
  Fact 0.86: In Nepal, the Cow (Bos taurus indicus) is listed as the national animal.
  Fact 0.87: In Nepal, the Danphe (Lophophorus impejanus) is listed as the national bird.
  Fact 0.88: In the Netherlands, the Lion (Panthera leo) is listed as the heraldic symbol.
  Fact 0.89: In the Netherlands, the Black-tailed godwit (Limosa limosa) is listed as the national bird.
  Fact 0.90: In New Zealand, the Kiwi (Apteryx sp.) is listed.
  Fact 0.91: In North Korea, the Chollima is listed as a mythical animal.
  Fact 0.92: In Norway, the Lion (Panthera leo) is listed as the royal national animal.
  Fact 0.93: In Norway, the White-throated dipper (Cinclus cinclus) is listed as the national bird.
  Fact 0.94: In Norway, the Fjord horse (Equus ferus caballus) is listed as the national horse.
  Fact 0.95: In Oman, the Arabian oryx (Oryx leucoryx) is listed as the national animal.
  Fact 0.96: In Pakistan, the Markhor (Capra falconeri) is listed as the National animal.
  Fact 0.97: In Pakistan, the Indus river dolphin (Platanista indicus minor minor) is listed as the National aquatic mammal.
  Fact 0.98: In Pakistan, the Chukar (Alectoris chukar) is listed as the National bird.
  Fact 0.99: In Pakistan, the Indus crocodile (Crocodylus palustris) is listed as the National reptile.
  Fact 0.100: In Pakistan, the Shaheen falcon (Falco peregrinus peregrinator) is listed as the Heritage bird.
  Fact 0.101: In Pakistan, the Snow leopard (Panthera uncia) is listed as the National predator.
  Fact 0.102: In Panama, the Harpy eagle (Harpia harpyja) is listed.
  Fact 0.103: In Papua New Guinea, the Dugong (Dugong dugon) is listed as the national marine mammal.
  Fact 0.104: In Peru, the Vicuña (Vicugna vicugna) is listed as the national animal.
  Fact 0.105: In the Philippines, the Carabao (Bubalus bubalis) is listed as the national animal.
  Fact 0.106: In the Philippines, the Philippine eagle (Pithecophaga jefferyi) is listed as the national bird.
  Fact 0.107: In Poland, the White-tailed eagle (Haliaeetus albicilla) is listed as the national bird.
  Fact 0.108: In Poland, the White stork (Ciconia ciconia) is listed as the national bird.
  Fact 0.109: In Poland, the European bison (Bison bonasus) is listed as the national animal.
  Fact 0.110: In Qatar, the Arabian oryx (Oryx leucoryx) is listed.
  Fact 0.111: In Romania, the Lynx (Lynx lynx) is listed.
  Fact 0.112: In Russia, the Eurasian brown bear (Ursus arctos arctos) is listed.
  Fact 0.113: In Rwanda, the Leopard (Panthera pardus) is listed.
  Fact 0.114: In Saint Vincent and the Grenadines, the Saint Vincent amazon (Amazona guildingii) is listed.
  Fact 0.115: In Serbia, the Wolf (Canis lupus lupus) is listed.
  Fact 0.116: In Serbia, the White eagle is listed as a mythical animal.
  Fact 0.117: In Serbia, the Lynx (Lynx lynx) is listed.
  Fact 0.118: In Somalia, the Leopard (Panthera pardus) is listed as the national animal.
  Fact 0.119: In South Africa, the Springbok (Antidorcas marsupialis) is listed as the national animal.
  Fact 0.120: In South Africa, the Blue crane (Anthropoides paradiseus) is listed as the national bird.
  Fact 0.121: In South Africa, the Galjoen (Dichistius capensis) is listed as the national fish.
  Fact 0.122: In South Korea, the Siberian tiger (Panthera tigris altaica) is listed.
  Fact 0.123: In South Korea, the Korean magpie (Pica (pica) sericea) is listed as the national bird.
  Fact 0.124: In Sri Lanka, the Sri Lankan junglefowl (Gallus lafayettii) is listed as the national bird.
  Fact 0.125: In Tanzania, the Giraffe (Giraffa camelopardalis) is listed.
  Fact 0.126: In Thailand, the Elephant (Elephas maximus) is listed.
  Fact 0.127: In the United Arab Emirates, the Arabian oryx (Oryx leucoryx) is listed as the national animal.
  Fact 0.128: In the United Arab Emirates, the Peregrine falcon (Falco peregrinus) is listed as the national bird.
  Fact 0.129: In the United Kingdom, the Lion (Panthera leo) is listed for the United Kingdom, England, Scotland and Wales.
  Fact 0.130: In the United Kingdom, the Bulldog (Canis lupus familiaris) is listed as Britain, called the British Bulldog.
  Fact 0.131: In the United Kingdom, the Red kite (Milvus milvus) is listed for Wales.
  Fact 0.132: In the United Kingdom, the Unicorn is listed as a mythical animal for Scotland.
  Fact 0.133: In the United Kingdom, the Y Ddraig Goch (Welsh Dragon) is listed as a mythical animal for Wales.
  Fact 0.134: In the United States, the Bald eagle (Haliaeetus leucocephalus) is listed as the national bird.
  Fact 0.135: In the United States, the American bison (Bison bison) is listed as the national mammal.

Retrieved items:
<item 1>
List of national animals
List of National Animals - wikipedia List of National Animals Jump to : navigation , search This article is incomplete . Please help to improve it , or discuss
the issue on the talk page . ( June 2017 ) This is a list of national animals . National animals Country Name of animal Scientific name Pictures Ref . Algeria
Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer
( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous
hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus
malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird
) Copsychus saularis Ilish ( national fish )
</item 1>

<item 2>
List of national animals
Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo
Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk
Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty
animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese
dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia
Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian
manatee ( national aquatic animal ) Trichechus manatus Croatia Pine
</item 2>

<item 3>
List of national animals
Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle (
national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx
leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent
amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa
Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South
Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus
lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant
</item 3>

<item 4>
List of national animals
Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer
Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator )
Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National
aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal )
Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG )
Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail (
</item 4>

<item 5>
List of national animals
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia
bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius
indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national
animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal
) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod )
Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus
impejanus Netherlands Lion ( heraldic symbol
</item 5>

<item 6>
List of national animals
Canadian Heritage . 2016 . Retrieved 1 December 2016 . Jump up ^ `` National Symbols of China '' . 123independenceday.com . Jump up ^ `` China National Symbols
'' . worldatlast.com. 2015 . Retrieved 2017 - 03 - 28 . Jump up ^ `` China Considers Red - crowned Crane for National Bird '' . www.china.org.cn . Jump up ^ ``
El cóndor , ave nacional - banrepcultural.org '' . Jump up ^ `` 16 de diciembre : Declaración del yigüirro como ave nacional de Costa Rica '' . Jump up ^ `` El
Venado Cola Blanca , Símbolo Nacional de la Fauna Silvestre '' . Jump up ^ `` Manatee Now Costa Rica 's New National Symbol '' . QCOSTARICA . Archived from the
original on 14 September 2014 . Jump up ^ Bousfield , Jonathan ( 2003 ) . Croatia . Rough Guides . p. 72 . ISBN 1843530848 .
</item 6>

<item 7>
List of national animals
bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx
leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain
as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical
United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison See also Animals portal Lists portal State
animal List of national birds Animals as heraldic charges Floral emblem National personification References Jump up ^ `` The Fennec Fox : Algeria 's National
Animal '' . algeria.com . Retrieved 2013 - 03 - 21 . Jump up ^ `` THE NATIONAL BIRD OF ANGOLA '' . BIRDS ANGOLA . Retrieved 12 April 2016 .
</item 7>

<item 8>
List of national animals
Jump up ^ Jon Eldridge . `` Primeval Poland : where the bison roam '' . theguardian.com . Retrieved 2016 - 10 - 10 . Jump up ^ `` Poland '' . allaboutbison.com
. Retrieved 2016 - 10 - 10 . Jump up ^ Tamra Orr ( 30 June 2008 ) . Qatar . Marshall Cavendish . p. 13 . ISBN 978 - 0 - 7614 - 2566 - 3 . Retrieved 30 July 2011
. Jump up ^ `` Animalul national - Rasul - Sapte simboluri nationale ale Romaniei mai putin cunoscute '' . 9AM . Jump up ^ `` Birobidjan , Russia Beaverton 's
sister city since 1990 '' . The City of Beaverton . Retrieved 2017 - 01 - 23 . Jump up ^ `` Rwanda State Symbols , Song , Flag , and More '' . worldatlas.com .
Retrieved 2013 - 03 - 21 .
</item 8>

<item 9>
List of national animals
Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New
Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus
cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra
falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National
reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy
eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao (
national animal ) Bubalus bubalis Philippine
</item 9>

<item 10>
Graph query row:
animal: Giant panda
country: China
relationship: RELATED_TO
fact: None
relationship_detail: None
</item 10>
=== END ITEM id=139996cd6d820173 ===

=== ITEM id=18f0e3ceee3fbcca ===
Gold sentences and facts:
Sentence 0: This article covers the professional life of George W. Bush , the 43rd President of the United States . Prior to his election as President in 2000 , Bush held numerous other positions , including being an oil executive , an owner of the Texas Rangers baseball team , and Governor of Texas . ( Also see Governorship of George W. Bush . )
  Fact 0.0: This article covers the professional life of George W. Bush.
  Fact 0.1: George W. Bush was the 43rd President of the United States.
  Fact 0.2: Bush was elected President in 2000.
  Fact 0.3: Before his election as President, Bush held numerous other positions.
  Fact 0.4: Bush was an oil executive.
  Fact 0.5: Bush was an owner of the Texas Rangers baseball team.
  Fact 0.6: Bush was Governor of Texas.
  Fact 0.7: The article Governorship of George W. Bush is a see-also reference.

Retrieved items:
<item 1>
Professional life of George W. Bush
attorneys controversy Email controversy Judicial appointments Supreme Court controversies Cabinet Pardons Impeachment efforts Life Presidential library Early
life Military service controversy ( Killian documents controversy and authenticity issues ) Professional life Governorship of Texas Prairie Chapel Ranch Bush
compound Clinton Bush Haiti Fund Speeches Axis of evil Mission Accomplished State of the Union address 2002 2003 2005 2006 2007 2008 Elections United States
House of Representatives elections , 1978 Texas gubernatorial election 1994 1998 Presidential campaign 2000 Republican Party presidential primaries , 2000
Republican National Convention 2000 United States presidential election , 2000 Bush v. Gore United States presidential election , 2004 Public image Bushisms
Nicknames As the subject of books and films Fictionalized portrayals Miss Me Yet ? Books A Charge to Keep ( 1999 ) Decision Points ( 2010 ) 41 : A Portrait of
My Father ( 2014 ) Portraits of Courage ( 2017 ) Family Laura Bush ( wife ) Barbara Pierce Bush ( daughter ) Jenna Bush Hager ( daughter ) George H.W.
</item 1>

<item 2>
Professional life of George W. Bush
Against the advice of his counsel , Bush repaid the loan by selling $848,000 worth of stock in Harken Energy . Harken reported significant financial losses
within a year of this sale , triggering allegations of insider trading . On March 27 , 1992 , the Securities and Exchange Commission concluded that Bush had a
`` preexisting plan '' to sell , that Bush had a `` relatively limited role in Harken management '' , and that it had not seen evidence of insider trading . The
subsequent SEC investigation ended in 1992 with a memo stating `` it appears that Bush did not engage in illegal insider trading , '' but noted that the memo ``
must in no way be construed as indicating that the party has been exonerated or that no action may ultimately result '' . Critics allege that this decision was
strongly influenced by the makeup of the SEC at the time , which heavily favored Bush . The chairman at the time was Richard Breeden , a good friend of the Bush
family 's who had been nominated to the SEC by President George H.W. Bush and who had been a lawyer in James Baker 's firm , Baker Botts .
</item 2>

<item 3>
Professional life of George W. Bush
In 1984 , Bush sold the company , hurt in the wake of the 1979 energy crisis , to Spectrum 7 , another Texas gas exploration firm . Under the terms of the sale
, Bush became CEO . Spectrum 7 lost revenue and was merged into Harken Energy Corporation in 1986 , with Bush becoming a director of Harken . Texas Rangers and
allegations of insider trading ( edit ) After working on his father 's successful 1988 presidential campaign , Bush learned from fellow Yale alumnus William
DeWitt , Jr. , that family friend Eddie Chiles wanted to sell the Texas Rangers baseball franchise along with the new sports stadium ; built on land acquired
under eminent domain law and built under funding financed through taxpayers ' funds backed by a bond issued for its debt . In April 1989 , Bush assembled a
group of investors from his father 's close friends , including fellow fraternity brother Roland W. Betts ; the group bought an 86 % share of the Rangers for
$75 million . Bush received a 2 % share by investing $606,302 , of which $500,000 was a bank loan .
</item 3>

<item 4>
Professional life of George W. Bush
Bush and who had been a lawyer in James Baker 's firm , Baker Botts . The SEC 's general counsel at the time was James Doty , who had been appointed by
President H.W. Bush and as a lawyer in James Baker 's firm , Baker Botts had represented George W. Bush when arrangements were made to acquire the Texas Rangers
baseball franchise ( although Doty recused himself from the investigation . ) . With Baker Botts representing W. Bush , the Saudi BinLaden family and W. Bush 's
funding conduit James R. Bath , Doty was involved in the frivilous litigation campaign launched in the attempt to intimidate BinLaden middleman James R Bath 's
business partner ( ( Charles W. ( `` Bill '' ) White ) ) into cooperating with the attempted cover - up of secret BinLaden Family funding of W. Bush 's
campaigns and businesses . Bush 's own lawyer was Robert Jordan , who had been `` partners with both Doty and Breeden at Baker Botts and who later became George
W. Bush 's ambassador to Saudi Arabia '' .
</item 4>

<item 5>
Professional life of George W. Bush
Jump up ^ `` CNN.com - White House defends Bush handling of stock sale - July 3 , 2002 '' . Jump up ^ `` Files : Bush Knew Firm 's Plight Before Stock Sale '' .
Washington Post . 2002 - 07 - 21 . Retrieved 2007 - 01 - 02 . Jump up ^ http://www.austinchronicle.com/issues/vol17/issue19/pols.bush.html Jump up ^ `` Team
Player - Texas - Joe Nick Patoski '' . Jump up ^ `` ESPN.com - Major League Baseball - Friday : Bush gets to know Texas '' . Jump up ^ `` Bush Was n't Always a
Front - Runner '' . Washington Post . 1999 - 10 - 17 . Retrieved 2007 - 04 - 07 .
</item 5>

<item 6>
Professional life of George W. Bush
Professional life of George W. Bush - wikipedia Professional life of George W. Bush Jump to : navigation , search George W. Bush This article covers the
professional life of George W. Bush , the 43rd President of the United States . Prior to his election as President in 2000 , Bush held numerous other positions
, including being an oil executive , an owner of the Texas Rangers baseball team , and Governor of Texas . ( Also see Governorship of George W. Bush . )
Contents ( hide ) 1 Business 1.1 Texas Rangers and allegations of insider trading 1.2 Early political career 2 See also 3 References Business ( edit ) Bush
began his industry career in 1979 , when he established Arbusto Energy , an oil and gas exploration company he financed with his education trust fund surplus
and money from other investors , including Dorothy Bush , Lewis Lehrman , William Henry Draper III , Bill Gammell , and James R. Bath , the last of whom
represented Salem bin Laden , a half - brother and cousin of Osama bin Laden .
</item 6>

<item 7>
Professional life of George W. Bush
In House of Bush , House of Saud , Craig Unger notes that at the time of Bush 's sale , Harken Energy `` was expected to run out of money in just three days ''
( p. 123 ) . In a last - ditch attempt to save the company , Harken was advised by the endowment fund of Harvard University to spin off two of its lower -
performing divisions . `` According to a Harken memo , if the plan did not go through , the company had ' no other source of immediate financing . ' '' Bush had
already taken out a $500,000 loan and sought Harken 's general counsel for advice . The reply was explicit : `` The act of trading , particularly if close in
time to the receipt of the inside information , is strong evidence that the insider 's investment decision was based on the inside information ... the insider
should be advised not to sell '' . This memo was turned over by Bush 's attorney the day after the U.S. Securities and Exchange Commission ( SEC ) ruled that it
would not charge Bush with insider trading .
</item 7>

<item 8>
Professional life of George W. Bush
Securities and Exchange Commission ( SEC ) ruled that it would not charge Bush with insider trading . On June 22 , Bush sold his 212,140 shares of stock anyway
for a net profit of $848,560 . The very next quarter , Harken announced losses of $23 million , which continued to the end of the year when the stock ``
plummeted from $4 to $1.25 '' . As President , Bush has refused to authorize the SEC to release its full report on the Harken investigation . When the Rangers
franchise was sold for $250 million in 1998 , at a total profit of $170 million , Bush personally received $14.9 million for his $600,000 investment . As
managing general partner of the Rangers , Bush assisted the team 's media relations and the construction of a new stadium . Many in Arlington protested that the
stadium was paid for with public funds , and when the stadium title was given to the Texas Rangers Organization , it effectively allowed Bush to cash out public
funds . His public role generated valuable goodwill and reinforced name recognition throughout Texas .
</item 8>

<item 9>
Graph query row:
connection: ['George W. Bush', 'Texas Rangers', None, 'Washington Senators move to Dallas - Ft. Worth (Arlington) and are renamed Texas Rangers']
relationship_details: [{'context': None, 'fact': None, 'relation': None, 'relationship': 'owner', 'role': None, 'type': 'RELATED_TO'}, {'context': None, 'fact':
None, 'relation': None, 'relationship': None, 'role': None, 'type': 'FROM_CHUNK'}, {'context': None, 'fact': None, 'relation': None, 'relationship': None,
'role': None, 'type': 'FROM_CHUNK'}]
ownership_mentioned: True
bush_family_or_george_hw_bush_mentioned: False
</item 9>

<item 10>
Professional life of George W. Bush
His public role generated valuable goodwill and reinforced name recognition throughout Texas . Early political career ( edit ) After a United States National
Guard transfer in 1972 , he served as political director for an Alabama senate campaign . In 1978 , Bush ran for the U.S. House of Representatives but lost to
Kent Hance . Bush stressed his energy credentials and conservative values in the campaign . Hance , however , also held many conservative views , opposing gun
control and strict regulation ; he successfully portrayed Bush as out of touch with rural Texans . Hance later became a Republican and donated money to Bush 's
campaign for Governor of Texas in 1993 . See also ( edit ) George W. Bush -- main article George W. Bush presidential campaign , 2000 References ( edit ) Jump
up ^ http://www.publicintegrity.org/docs/harken/harken_doc5.pdf Jump up ^ `` White House Watch : Credibility problem '' . UPI. 17 July 2002 . Jump up ^
http://www.publicintegrity.org/docs/harken/harken_doc7.pdf Jump up ^ `` Washingtonpost.com : Bush Name Helps Fuel Oil Dealings '' .
</item 10>
=== END ITEM id=18f0e3ceee3fbcca ===

=== ITEM id=23bc95abd99453d8 ===
Gold sentences and facts:
Sentence 0: Dollree Mapp was an employee in the illegal gambling rackets dominated by Cleveland rackets kingpin Shon Birns . On May 23 , 1957 , police officers in Cleveland , Ohio , received an anonymous tip by phone that Virgil Ogletree , a numbers operator who was wanted for questioning in the bombing of rival numbers racketeer and future boxing promoter Don King 's home three days earlier , might be found at Mapp 's house , as well as illegal betting slips and equipment employed in the `` California Gold '' numbers operation set up by Mapp 's boyfriend Edward Keeling . Three officers went to the home and asked for permission to enter , but Mapp , after consulting her lawyer by telephone , refused to admit them without a search warrant . Two officers left , and one remained , watching the house from across the street .
  Fact 0.0: Dollree Mapp was an employee in the illegal gambling rackets.
  Fact 0.1: The illegal gambling rackets were dominated by Cleveland rackets kingpin Shon Birns.
  Fact 0.2: On May 23, 1957, police officers in Cleveland, Ohio, received an anonymous tip by phone.
  Fact 0.3: The anonymous tip said Virgil Ogletree might be found at Dollree Mapp's house.
  Fact 0.4: Virgil Ogletree was a numbers operator.
  Fact 0.5: Virgil Ogletree was wanted for questioning in the bombing of Don King's home three days before the tip.
  Fact 0.6: Don King was a rival numbers racketeer.
  Fact 0.7: Don King was a future boxing promoter.
  Fact 0.8: The anonymous tip said illegal betting slips might be found at Dollree Mapp's house.
  Fact 0.9: The anonymous tip said equipment employed in the "California Gold" numbers operation might be found at Dollree Mapp's house.
  Fact 0.10: The "California Gold" numbers operation was set up by Dollree Mapp's boyfriend Edward Keeling.
  Fact 0.11: Edward Keeling was Dollree Mapp's boyfriend.
  Fact 0.12: Three officers went to Dollree Mapp's home.
  Fact 0.13: The three officers asked for permission to enter the home.
  Fact 0.14: Dollree Mapp consulted her lawyer by telephone.
  Fact 0.15: Dollree Mapp refused to admit the officers without a search warrant.
  Fact 0.16: Two officers left.
  Fact 0.17: One officer remained, watching the house from across the street.

Retrieved items:
<item 1>
Mapp v. Ohio
§ 1983 Monroe v. Pape ( 1961 ) Malley v. Briggs ( 1986 ) ( hide ) Incorporation against States Unreasonable search and seizure Wolf v. Colorado ( 1949 ) Rochin
v. California ( 1952 ) Irvine v. California ( 1954 ) Elkins v. United States ( 1960 ) Mapp v. Ohio ( 1961 ) Warrant requirements Ker v. California ( 1963 )
Aguilar v. Texas ( 1964 ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Mapp_v._Ohio&oldid=819300745 '' Categories : United States Supreme Court
decisions that overrule a prior Supreme Court decision Cuyahoga County , Ohio United States Supreme Court cases United States Fourth Amendment case law 20th
century American trials 1961 in United States case law Incorporation case law American Civil Liberties Union litigation United States Supreme Court cases of the
Warren Court Cleveland Division of Police Hidden categories : Articles needing additional references from October 2017 All articles needing additional
references Talk Contents About Wikipedia Simple English Edit links This page was last edited on 8 January 2018 , at 16 : 31 .
</item 1>

<item 2>
Mapp v. Ohio
Supreme Court voted 5 - 3 in favor of Mapp . The Court overturned the conviction , and five justices found that the States were bound to exclude evidence seized
in violation of the 4th Amendment . This ruling officially applied the exclusionary rule to the states . See also ( edit ) United States Bill of Rights Fruit of
the poisonous tree Exclusionary rule References ( edit ) Jump up ^ Zotti , Priscilla . Injustice for All ( Peter Lang , 2005 ) . Jump up ^ Mapp v. Ohio , 367
U.S. 643 ( U.S. 1961 ) . Jump up ^ Duignan , Brian ( 2012 - 05 - 25 ) . `` Mapp v. Ohio '' . Encyclopædia Britannica Online . Encyclopædia Britannica Inc .
Retrieved March 19 , 2014 . Jump up ^ https://www.law.cornell.edu/supremecourt/text/367/643#writing - Jump up ^ Williams , Bob . `` Dolly Mapp Sheds Glamour Gal
Role '' , Cleveland Call and Post , Nov.
</item 2>

<item 3>
Mapp v. Ohio
23 , 1957 , p. 1 . Jump up ^ http://www.infoplease.com/us/supreme-court/cases/ar19.html Further reading ( edit ) Long , Carolyn ( 2006 ) . Mapp v. Ohio :
Guarding Against Unreasonable Searches and Seizures . University Press of Kansas . ISBN 0 - 7006 - 1441 - 9 . Stewart , Potter ( 1983 ) . `` The Road to Mapp v.
Ohio and beyond : The Origins , Development and Future of the Exclusionary Rule in Search - and - Seizure Cases '' . Columbia Law Review . 83 ( 6 ) : 1365 --
1404 . doi : 10.2307 / 1122492 . JSTOR 1122492 . Zotti , Priscilla H. Machado ( 2005 ) . Injustice for All : Mapp vs. Ohio and the Fourth Amendment . New York :
Peter Lang . ISBN 0 - 8204 - 7267 - 0 .
</item 3>

<item 4>
Mapp v. Ohio
Ohio Supreme Court reversed . Court membership Chief Justice Earl Warren Associate Justices Hugo Black Felix Frankfurter William O. Douglas Tom C. Clark John M.
Harlan II William J. Brennan , Jr . Charles E. Whittaker Potter Stewart Case opinions Majority Clark , joined by Warren , Black , Douglas , Brennan Concurrence
Black Concurrence Douglas Concurrence Stewart Dissent Harlan , joined by Frankfurter , Whittaker Laws applied U.S. Const . amends. IV , XIV This case overturned
a previous ruling or rulings Wolf v. Colorado ( 1949 ) Mapp v. Ohio , 367 U.S. 643 ( 1961 ) , was a landmark case in criminal procedure , in which the United
States Supreme Court decided that evidence obtained in violation of the Fourth Amendment , which protects against `` unreasonable searches and seizures , '' may
not be used in state law criminal prosecutions in state courts , as well as in federal criminal law prosecutions in federal courts as had previously been the
law .
</item 4>

<item 5>
Mapp v. Ohio
Mapp was arrested , charged , and cleared on a misdemeanor charge of possessing numbers paraphernalia ; but several months later , after she refused to testify
against Shon Birns , Edward Keeling and their associates at their trial that October for the attempted shakedown of Don King , she was prosecuted for possession
of the books , found guilty at a 1958 trial of `` knowingly having had in her possession and under her control certain lewd and lascivious books , pictures ,
and photographs in violation of 2905.34 of Ohio 's Revised Code '' , and sentenced to one to seven years in prison . Mapp then appealed her case to the Supreme
Court , on the grounds that the police had no probable cause to suspect her of having the books . She stated that the 4th Amendment should be incorporated to
the state and local level . She argued that the police could n't use the books as evidence in trial because they were found without a warrant and therefore
illegally recovered . Mapp is the first to call upon this law . Decision ( edit ) The U.S. Supreme Court voted 5 - 3 in favor of Mapp .
</item 5>

<item 6>
Mapp v. Ohio
The Supreme Court accomplished this by use of a principle known as selective incorporation ; in Mapp this involved the incorporation of the provisions , as
interpreted by the Court , of the Fourth Amendment which are applicable only to actions of the federal government into the Fourteenth Amendment due process
clause which is applicable to actions of the states . Contents ( hide ) 1 Circumstances of the case 2 Decision 3 See also 4 References 5 Further reading 6
External links Circumstances of the case ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( October 2017 ) ( Learn how and when to remove this template message )
Dollree Mapp was an employee in the illegal gambling rackets dominated by Cleveland rackets kingpin Shon Birns .
</item 6>

<item 7>
Mapp v. Ohio
On May 23 , 1957 , police officers in Cleveland , Ohio , received an anonymous tip by phone that Virgil Ogletree , a numbers operator who was wanted for
questioning in the bombing of rival numbers racketeer and future boxing promoter Don King 's home three days earlier , might be found at Mapp 's house , as well
as illegal betting slips and equipment employed in the `` California Gold '' numbers operation set up by Mapp 's boyfriend Edward Keeling . Three officers went
to the home and asked for permission to enter , but Mapp , after consulting her lawyer by telephone , refused to admit them without a search warrant . Two
officers left , and one remained , watching the house from across the street . Thirteen hours later , four cars full of police arrived and knocked on the door .
When she did n't answer , they forced the door . Mapp asked to see the alleged warrant and was shown a piece of paper which she snatched away from an officer ,
putting it inside her dress . The officers struggled with Mapp and recovered the piece of paper which was not seen by her or her lawyers again , and was not
introduced as evidence in any of the ensuing court proceedings .
</item 7>

<item 8>
Mapp v. Ohio
When asked about the warrant during oral argument at the Supreme Court , the Cleveland prosecutor arguing the case cautiously deflected the question , which the
court did not press . As the search of Mapp 's second - floor , 2 - bedroom apartment began , police handcuffed her for being belligerent . The police searched
the house thoroughly , and discovered Ogletree , who was subsequently cleared on the bombing charge , hiding in the apartment of the downstairs tenant , Minerva
Tate . In the search of Mapp 's apartment and in a footlocker in the basement of the house police found a quantity of `` California Gold '' betting slips and
paraphernalia . They also found a pistol and a small quantity of pornographic books and pictures which Mapp stated a previous tenant named Morris Jones had left
behind .
</item 8>

<item 9>
Mapp v. Ohio
ISBN 0 - 8204 - 7267 - 0 . External links ( edit ) Wikisource has original text related to this article : Mapp v. Ohio Archival source documents relating to the
Mapp case at Cleveland Memory Dollree Mapp , Who Defied Police Search in Landmark Case , Is Dead - New York Times `` Supreme Court Landmark Case Mapp v. Ohio ''
from C - SPAN 's Landmark Cases : 12 Historic Supreme Court Decisions ( hide ) United States 4th Amendment case law Scope of the Fourth Amendment What
constitutes a search ? Ex parte Jackson ( 1878 ) Boyd v. United States ( 1886 ) Gouled v. United States ( 1921 ) Burdeau v. McDowell ( 1921 ) Hester v. United
States ( 1924 ) United States v. Lee ( 1927 ) Olmstead v. United States ( 1928 ) Goldman v. United States ( 1942 ) On Lee v. United States ( 1952 ) Abel v.
United States ( 1960 ) Silverman v. United States ( 1961 ) Lewis v.
</item 9>

<item 10>
Mapp v. Ohio
Mapp v. Ohio - wikipedia Mapp v. Ohio Jump to : navigation , search Mapp v. Ohio Supreme Court of the United States Argued March 29 , 1961 Decided June 19 ,
1961 Full case name Dollree Mapp v. State of Ohio Citations 367 U.S. 643 ( more ) 81 S. Ct. 1684 ; 6 L. Ed . 2d 1081 ; 1961 U.S. LEXIS 812 ; 86 Ohio L. Abs. 513
; 16 Ohio Op . 2d 384 ; 84 A.L.R. 2d 933 Prior history Defendant convicted , Cuyahoga County , Ohio Court of Common Pleas ; affirmed , Ohio Court of Appeals ;
affirmed , Ohio Supreme Court 166 N.E. 2d 387 ( Ohio 1960 ) Subsequent history Rehearing denied , 368 U.S. 871 ( 1961 ) Holding The Fourth Amendment prohibition
against unreasonable searches and seizures , as applied to the states through the Fourteenth , excludes unconstitutionally obtained evidence from use in
criminal prosecutions . Ohio Supreme Court reversed .
</item 10>
=== END ITEM id=23bc95abd99453d8 ===

=== ITEM id=93c956d73bdd3488 ===
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
Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say
Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' (
Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 (
From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by
Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by
The Cox Family ) `` I Will '' ( John Lennon ,
</item 2>

<item 3>
*Now That I’ve Found You: A Collection* is Alison Krauss’s retrospective compilation of her early recording career. Released by Rounder Records on February 7,
1995, it features bluegrass and country music and runs 42 minutes, 31 seconds. Produced by Krauss and Union Station, it followed *I Know Who Holds Tomorrow*
(1994) and preceded *So Long So Wrong* (1997). Its 1995 singles were “Baby, Now That I
</item 3>

<item 4>
Now That I've Found You: A Collection
1995 Succeeded by Starting Over by Reba McEntire ( hide ) Alison Krauss & Union Station Alison Krauss Barry Bales Ron Block Jerry Douglas Dan Tyminski John
Pennell Jeff White Brent Truitt Mike Harman Adam Steffey Tim Stafford Studio albums Different Strokes Too Late to Cry Two Highways I 've Got That Old Feeling
Every Time You Say Goodbye I Know Who Holds Tomorrow So Long So Wrong Forget About It New Favorite Lonely Runs Both Ways Paper Airplane Windy City Compilation
albums Now That I 've Found You : A Collection Live A Hundred Miles or More : A Collection Essential Alison Krauss With Robert Plant Raising Sand Singles ``
When You Say Nothing at All '' `` Baby Now That I 've Found You '' `` Baby Mine '' `` The Lucky One '' `` Missing You '' `` Please Read the Letter '' `` Rich
Woman '' Featured singles `` Teach Your Children '' `` Somewhere in the Vicinity of the Heart '' `` It 's Not Over '' `` Buy Me a Rose '' `` Whiskey Lullaby ''
Related articles Discography Awards and nominations Alison Brown Robert Plant Viktor Krauss Up !
</item 4>

<item 5>
Now That I've Found You: A Collection
Now That I 've Found You : a Collection - wikipedia Now That I 've Found You : a Collection Jump to : navigation , search Now That I 've Found You : A
Collection Compilation album by Alison Krauss Released February 7 , 1995 Genre Bluegrass / Country Length 42 : 31 Label Rounder Producer Alison Krauss & Union
Station Alison Krauss chronology I Know Who Holds Tomorrow ( 1994 ) I Know Who Holds Tomorrow 1994 Now That I 've Found You : A Collection ( 1995 ) So Long So
Wrong ( 1997 ) So Long So Wrong 1997 Singles from Now That I 've Found You : A Collection `` Baby , Now That I 've Found You '' Released : 1995 `` Oh , Atlanta
'' Released : 1995 Professional ratings Review scores Source Rating Allmusic Robert Christgau A − Rolling Stone Now That I 've Found You : A Collection is an
album by Alison Krauss , released February 7 , 1995 . It is a retrospective of the early part of Krauss ' recording career .
</item 5>

<item 6>
Now That I've Found You: A Collection
It is a retrospective of the early part of Krauss ' recording career . It includes songs that appeared on her solo albums , albums by Alison Krauss & Union
Station , and some that appeared on an album by Alison Krauss & the Cox Family . It also contains some new material . The album features the hit country single
`` When You Say Nothing at All '' , sold over two million copies earning double platinum RIAA certification . The title single `` Baby , Now That I 've Found
You '' won the Grammy Award for Best Female Country Vocal Performance and `` When You Say Nothing at All '' was named Single of the Year by the Country Music
Association . Contents ( hide ) 1 Track listing 2 Personnel 3 Chart performance 3.1 Album 4 References Track listing ( edit ) `` Baby , Now That I 've Found You
'' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh ,
</item 6>

<item 7>
Now That I've Found You: A Collection
Close and Personal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Now_That_I%27ve_Found_You:_A_Collection&oldid=808197945 '' Categories : 1995
compilation albums Alison Krauss & Union Station albums Rounder Records compilation albums Hidden categories : Articles with hAudio microformats Music infoboxes
with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 1 November 2017 , at 13 : 34 . About Wikipedia
</item 7>

<item 8>
The excerpts cover several country releases. Kane Brown and Lauren Alaina’s 2017 duet “What Ifs,” about relationship uncertainty, topped Billboard’s Hot Country
Songs chart, ending Sam Hunt’s 34-week run. Alison Krauss’s 1995 *Now That I’ve Found You: A Collection* compiled early solo, Union Station and Cox Family
recordings with two unreleased songs; “When You Say Nothing at All” sold over two million copies, and the title track won a Grammy
</item 8>

<item 9>
*Now That I’ve Found You: A Collection* is an Alison Krauss compilation featuring two previously unreleased songs, “Now That I’ve Found You” and “Broadway,”
alongside recordings from her albums and collaborations, including “When You Say Nothing at All.” Krauss sings and plays fiddle and viola; contributors include
Ron Block, Jerry Douglas, Sam Bush, Stuart Duncan and others. The album reached No. 1 on Canada’s RPM Country Albums chart in 1995 and
</item 9>

<item 10>
*Now That I’ve Found You: A Collection* is a retrospective of Alison Krauss’s early recording career, drawing songs from her solo albums, work with Union
Station and the Cox Family, plus previously unreleased material. It features “When You Say Nothing at All,” which sold over two million copies and earned
double-platinum certification. The title track won Krauss a Grammy for Best Female Country Vocal Performance, and “When You Say Nothing at All” was named CMA
Single of the Year
</item 10>
=== END ITEM id=93c956d73bdd3488 ===
