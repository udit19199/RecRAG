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

=== ITEM id=9fe9e7fa1e4091f6 ===
Gold sentences and facts:
Sentence 0: Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995
  Fact 0.0: Mumbai has the Marathi name मुंबई.
  Fact 0.1: Mumbai was previously named Bombay.
  Fact 0.2: Mumbai was renamed in 1995.

Retrieved items:
<item 1>
Renaming of cities in India
In the early years after Indian independence , many name changes were effected in northern India for English spellings of Hindi place names that had simply been
romanised inconsistently by the British administration -- such as the British spelling `` Jubbulpore , '' renamed `` Jabalpur '' ( जबलपुर ) among the first
changes in 1947 . These changes did not generate significant controversy . More recent and high - profile changes -- including renaming such major cities as
Calcutta to Kolkata -- have generated greater controversy . Since independence , such changes have typically been enacted officially by legislation at local or
national Indian government level , and may or may not then be adopted by the Indian media , particularly the influential Indian press . In the case of smaller
towns and districts which were less notable outside and inside India , and where a well known English name ( or exonym ) could not be said to exist , older
spellings used under British India may not have had any specific legislation other than changes in practice on the romanisation of indigenous Indian language
names .
</item 1>

<item 2>
Renaming of cities in India
Jump up ^ Calcutta : A Cultural and Literary History - Page 3 Krishna Dutta - 2003 `` nationalist stance , like Bombay , which changed its name to Mumbai , or
Madras , which has become the unrecognisable Chennai , Calcutta has preferred a comparatively minor name change , which frankly is a bit of a multicultural
mishmash . '' Jump up ^ `` David Rumsey : Geographical Searching with MapRank Search ( beta ) '' . Rumsey.mapranksearch.com . Retrieved 2012 - 08 - 15 . Jump up
^ Beam , Christopher ( 2006 - 07 - 12 ) . `` How Bombay became Mumbai . - Slate Magazine '' . Slate.com . Retrieved 2012 - 08 - 15 .
</item 2>

<item 3>
Renaming of cities in India
The name changes are not without controversy among Indians . In several instances , the name change represents a struggle between a cosmopolitan elite and a
local , regional - language populace over defining the city in ways that go far beyond a simple change of name . '' Jump up ^ Steven I. Wilkinson Votes and
Violence : Electoral Competition and Ethnic Riots in India 2006 Page 23 `` The BJP proposed in 1990 and 2001 that Ahmedabad be renamed `` Karnavati . '' Hindu ,
June 1 1 , 2001 . Similar proposals have been made to rename Allahabad `` Prayag . '' Jump up ^ Cosmopolitanism - Page 73 Carol A. Breckenridge , Sheldon
Pollock , Homi K. Bhabha - 2002 `` In one sense , the decision to officialise the name Mumbai is part of a widespread Indian pattern of replacing names
associated with colonial rule with names associated with local , national , and regional heroes .
</item 3>

<item 4>
Renaming of cities in India
Madhya Bharat → Madhya Pradesh ( 1 November 1959 ) Madras State → Tamil Nadu ( 14 January 1969 ) Mysore State → Karnataka ( 1 November 1973 ) Uttaranchal →
Uttarakhand ( 1 January 2007 ) Orissa → Odisha ( 4 November 2011 ) Cities ( edit ) Notable city names that were officially changed by legislation after
independence include : Jabalpur ( Hindi : जबलपुर ) , from Jubbulpore , respelled in 1947 Jajmau ( Hindi : जाजमऊ ) , from Jajesmow , respelled in 1948 Kanpur (
Hindi : कानपुर ) , from Cawnpore , respelled in 1948 Vadodara ( Gujarati : વડોદરા ) , from Baroda , respelled in 1974 Thiruvananthapuram ( Malayalam :
തിരുവനന്തപുരം ) , from Trivandrum ,
</item 4>

<item 5>
Renaming of cities in India
from Trivandrum , respelled in 1991 Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995 Kochi ( Malayalam : കൊച്ചി ) , from Cochin , respelled in 1996
Chennai ( Tamil : சென்னை ) , from Madras , renamed in 1996 Kolkata ( Bengali : কলকাতা ) , from Calcutta , respelled in 2001 Kadapa ( Telugu : కడప ) , from
Cuddapah , respelled in 2005 Puducherry ( Tamil : புதுச்சேரி ) , from Pondicherry , renamed in 2006 Bengaluru ( Kannada : ಬೆಂಗಳೂರು ) , from Bangalore ,
</item 5>

<item 6>
Renaming of cities in India
Realignment of the official Indian English name to an alternative local name ( edit ) Aside from changes to the official English spellings of local names there
have also been renaming proposals to realign the official name , hence the English name with an alternative local name . Ethnically sensitive examples include
the proposals by the Bharatiya Janata Party ( 1990 , 2001 ) to rename Ahmedabad to Karnavati and Allahabad to Prayag . These two proposals are changes from the
Muslim ( or Mughal ) name to a Hindu native name . These can be represented as a change from Urdu language to Hindi language , but since the two languages are
variants of Hindustani the proposal is effectively a cultural and ethno - religious proposal rather than a linguistic one . Adoption of name changes in India
and overseas ( edit ) Official name changes naturally take place quickly if not immediately in official government sources . Adoption may be slower among the
media in India and abroad , and among Indian authors . Important examples ( edit ) Main article : List of renamed places in India States ( edit ) Travancore -
Cochin → Kerala ( 1 November 1956 ) .
</item 6>

<item 7>
Renaming of cities in India
Renaming of cities in India - Wikipedia Renaming of cities in India Jump to : navigation , search The renaming of cities in India started in 1947 following the
end of the British imperial period . Several changes were politically controversial , and not all proposed changes were implemented . Each had to be approved by
the central government in New Delhi . The renaming of states and territories in India has also taken place , but until recently with actual substantial name
changes in both local language and in English such as the old British state name of Travancore - Cochin to Kerala ( 1956 ) . The most notable recent exceptions
are Indian English spelling - changes of Orissa to Odisha ( March 2011 ) and the Union Territory of Pondicherry ( which includes the City of Pondicherry ) to
Puducherry .
</item 7>

<item 8>
Renaming of cities in India
For example , the change of Madras ( Tamil : மதராஸ் Madras ) to Chennai ( Tamil : சென்னை Chennai ) was reflected in many of India 's languages , and
incidentally in English , while the Tamil endonym had always been Chennai and remained unaffected by the change . Renaming in English ( edit ) Change in
official English spelling ( edit ) The renaming of cities is often specifically from English to Indian English in connection with that dialect 's internal
reforms . In other words , the city itself is not actually renamed in the local language , and the local name ( or endonym ) in the indigenous languages of
India does not change , but the official spelling in Indian English is amended . An example is the change from English `` Calcutta '' to English `` Kolkata ''
-- the local Bengali name ( কলকাতা Kôlkata ) did not change . Such changes in English spelling may be in order to better reflect a more accurate phonetic
transliteration of the local name , or may be for other reasons .
</item 8>

<item 9>
Renaming of cities in India
Jump up ^ `` Goa to become Govapuri '' . Jump up ^ `` Nagaland to be renamed '' . Aaj Tak . Jump up ^
http://www.dailypioneer.com/state-editions/lucknow/allahabad-all-set-to-get-new-name.html Jump up ^ TNN , Dec 18 , 2006 , 02.42 am IST ( 2006 - 12 - 18 ) . ``
Now , Indore to become Indur , Bhopal Bhojpal - The Times of India '' . The Times of India . Retrieved 2010 - 11 - 04 . CS1 maint : Multiple names : authors
list ( link ) Jump up ^ http://www.igovernment.in/opinion/41631/merging-npr-uid?source=igov External links ( edit ) Renaming roads : A ' meaningless ' exercise
12 Cities in Karnataka get a name change India 's Bangalore in name change Shashi Tharror : Becoming Bengaloorued Retrieved from ``
https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Renaming of cities in India
It is an indigenizing toponymic '' Jump up ^ Reserve Bank of India 's instructions for banks & banking operations Reserve Bank of India 2001 Page 713 `` The new
name `` Mumbai '' should be reflected in both English and Hindi and the change in name is to be brought about in all official communications , name plates ,
sign boards , office seals , rubber stamps , etc . '' Jump up ^ Perveez Mody The Intimate State : Love - Marriage and the Law in Delhi Page 59 - 2008 ``
Throughout this book , I refer to India 's commercial capital as Bombay rather than Mumbai ... I am well aware of the name - change effected by an Act of the
Indian Parliament in 1997 that made the city officially ' Mumbai ' ... It is the same convention I adopt when referring to Calcutta rather than Kolkata . ''
Jump up ^ Pingali Sailaja Indian English Page 16 2009 `` Bombay is now called Mumbai , Madras is now Chennai and Calcutta is Kolkata , in an attempt to
de-anglicise them . In this work , the earlier names are retained since these names were used during the period that we mostly cover . ''
</item 10>
=== END ITEM id=9fe9e7fa1e4091f6 ===

=== ITEM id=2fe1fdc02471aa93 ===
Gold sentences and facts:
Sentence 0: The concept was developed by Georges Auguste Escoffier . This structured team system delegates responsibilities to different individuals who specialize in certain tasks in the kitchen .
  Fact 0.0: The concept was developed by Georges Auguste Escoffier.
  Fact 0.1: The structured team system delegates responsibilities to different individuals.
  Fact 0.2: The individuals who receive delegated responsibilities specialize in certain tasks in the kitchen.

Retrieved items:
<item 1>
List of Marvel Comics characters: B
Roekel 's men acquired the bodies . The corpses were perfect specimens : intact and filled with the lethal bio-weapon gas mixture , their flesh was still alive
. Elements of the hundred man platoon were combined with a single consciousness - the consciousness of the former platoon 's commanding officer , Rick Landau .
Brigade is capable of firing a variety of plasma bolts / ammunition from his hands and shoulders . Using this power , combined with his high rate - of - fire he
can decimate even the strongest opponents quickly and with ease . But , being a work in progress , certain mental ' irregularities ' had not been perfected .
Brigade was prone to fits of dissociative identity seizures where any number of his former teammates ' personalities broke out , rejecting the singular identity
. Besides this , Brigade is the ultimate warrior . During the imperfect invasion of New York City , Brigade encountered and fought against Wolverine . He
defeated Wolverine and later on was toppled by Iron Man during a raid on Van Roekel headquarters where all the other mutant specimens were being held in
captivity .
</item 1>

<item 2>
The brigade de cuisine is a kitchen hierarchy developed by Georges Auguste Escoffier for restaurants and hotels with large staffs. It divides work among
specialists, though smaller kitchens combine roles. The chef de cuisine manages staff, menus, purchasing, training, and hygiene; the sous-chef assists. Station
chefs and cooks handle tasks such as sauces, roasting, fish, vegetables, pantry, and pastry. Other roles include apprentices, dishwashers, butchers, bakers, and
an exped
</item 2>

<item 3>
Brigade de cuisine
tempus n ° 359 ) , 2010 , 381 pages . ISBN 978 - 2 - 262 - 03318 - 7 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Brigade_de_cuisine&oldid=783032044 '' Categories : Food services occupations Restaurant terminology Culinary
terminology Hidden categories : Use dmy dates from April 2011 Talk Contents About Wikipedia Deutsch Español Français Italiano Magyar 日本 語 Shqip Edit links This
page was last edited on 30 May 2017 , at 18 : 15 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 3>

<item 4>
Brigade de cuisine
Confiseur in larger restaurants , prepares candies and petits fours instead of the pâtissier . Glacier in larger restaurants , prepares frozen and cold desserts
instead of the pâtissier . Décorateur in larger restaurants , prepares show pieces and specialty cakes instead of the pâtissier . Boulanger ( baker ) in larger
restaurants , prepares bread , cakes , and breakfast pastries instead of the pâtissier . Boucher ( butcher ) butchers meats , poultry , and sometimes fish ; may
also be in charge of breading meat and fish items . Aboyeur ( announcer / expediter ) takes orders from the dining room and distributes them to the various
stations ; may also be performed by the sous - chef de partie . Communard prepares the meal served to the restaurant staff . Garçon de cuisine ( literally ``
kitchen boy '' ) in larger restaurants , performs preparatory and auxiliary work for support . See also ( edit ) Food portal Auguste Escoffier Chef List of
restaurant terminology Maître d'hôtel , a front of house head Notes ( edit ) Jump up ^ `` Escoffier Kitchen Brigade System Then and Now '' .
</item 4>

<item 5>
Brigade de cuisine
Brigade de cuisine - wikipedia Brigade de cuisine Jump to : navigation , search Brigade de cuisine ( French : ( bʁiɡad də kɥizin ) , kitchen brigade ) is a
system of hierarchy found in restaurants and hotels employing extensive staff , commonly referred to as `` kitchen staff '' in English - speaking countries .
The concept was developed by Georges Auguste Escoffier . This structured team system delegates responsibilities to different individuals who specialize in
certain tasks in the kitchen . Contents ( hide ) 1 List of positions 2 See also 3 Notes 4 References List of positions ( edit ) This is an exhaustive list of
the different members of the kitchen brigade system . Only the largest of establishments would have an extensive staff of this size . As noted under some titles
, certain positions are combined into other positions when such a large staff is unnecessary . Note : Despite the use of chef in English as the title for a cook
, the word actually means `` chief '' or `` head '' in French . Similarly , cuisine means `` kitchen '' , but also refers to food or cooking generally , or a
type of food or cooking .
</item 5>

<item 6>
Brigade de cuisine
Friturier ( fry cook ) in larger kitchens , prepares fried foods instead of the rôtisseur . Poissonnier ( fish cook ) prepares fish and seafood dishes .
Entremetier ( entrée preparer ) prepares soups and other dishes not involving meat or fish , including vegetable dishes and egg dishes . Potager ( soup cook )
in larger kitchens , reports to the entremetier and prepares the soups . Legumier ( vegetable cook ) in larger kitchens , also reports to the entremetier and
prepares the vegetable dishes . Garde manger ( pantry supervisor ; literally `` food keeper '' ) is responsible for preparation of cold hors d'oeuvres , pâtés ,
terrines and aspics ; prepares salads ; organizes large buffet displays ; and prepares charcuterie items . Tournant ( spare hand / roundsman ) moves throughout
the kitchen , assisting other positions in kitchen . Pâtissier ( pastry cook ) prepares desserts and other meal - end sweets , and for locations without a
boulanger , also prepares breads and other baked items ; may also prepare pasta for the restaurant .
</item 6>

<item 7>
Brigade de cuisine
6 June 2014 . Retrieved 3 May 2017 . Jump up ^ `` The Kitchen Brigade - Lucky Peach '' . 28 October 2015 . Retrieved 3 May 2017 . ^ Jump up to : Dominé , 32 . ^
Jump up to : Dominé , 33 . ^ Jump up to : The Culinary Institute of America , 8 . References ( edit ) Dominé , André ( ed . ) . Culinaria France . Cologne :
Könemann Verlagsgesellschaft mbh , 1998 . ISBN 978 - 3 - 8331 - 1129 - 7 The Culinary Institute of America ( 2006 ) . The Professional Chef ( 8th ed . ) .
Hoboken , New Jersey : John Wiley & Sons . ISBN 978 - 0 - 7645 - 5734 - 7 . Patrick Rambourg , Histoire de la cuisine et de la gastronomie françaises , Paris ,
Ed . Perrin ( coll . tempus n ° 359 ) , 2010 , 381 pages .
</item 7>

<item 8>
Brigade de cuisine
Chef de cuisine ( kitchen chef ; literally `` chief of kitchen '' ) is responsible for overall management of kitchen ; supervises staff , creates menus and new
recipes with the assistance of the restaurant manager , makes purchases of raw food items , trains apprentices , and maintains a sanitary and hygienic
environment for the preparation of food . Sous - chef de cuisine ( deputy / second kitchen chef ; literally `` subchief '' ) receives orders directly from the
chef de cuisine for the management of the kitchen , and often serves as the representative when the chef de cuisine is not present . Saucier ( saucemaker /
sauté cook ) prepares sauces and warm hors d'oeuvres , completes meat dishes , and in smaller restaurants , may work on fish dishes and prepare sautéed items .
This is one of the most respected positions in the kitchen brigade . Chef de partie ( senior chef ; literally `` chief of party '' ; party used here as a group
, in the sense of a military detail ) is responsible for managing a given station in the kitchen , specializing in preparing particular dishes there . Those who
work in a lesser station are commonly referred to as a demi - chef .
</item 8>

<item 9>
Brigade de cuisine
Those who work in a lesser station are commonly referred to as a demi - chef . Cuisinier ( cook ) is an independent position , usually preparing specific dishes
in a station ; may also be referred to as a cuisinier de partie . Commis ( junior cook ) also works in a specific station , but reports directly to the chef de
partie and takes care of the tools for the station . Apprenti ( e ) ( apprentice ) are often students gaining theoretical and practical training in school and
work experience in the kitchen . They perform preparatory work and / or cleaning work . Plongeur ( dishwasher or kitchen porter ) cleans dishes and utensils ,
and may be entrusted with basic preparatory jobs . Marmiton ( pot and pan washer , also known as kitchen porter ) in larger restaurants , takes care of all the
pots and pans instead of the plongeur . Rôtisseur ( roast cook ) manages a team of cooks that roasts , broils , and deep fries dishes . Grillardin ( grill cook
) in larger kitchens , prepares grilled foods instead of the rôtisseur .
</item 9>

<item 10>
Under Louis XIV, dance became formalized: professional dancers replaced court amateurs, ballet masters were licensed, and the Académie Royale de Danse opened in
Paris in 1661. Its troupe began as all-male and admitted women in 1681. In the 20th century, modern dance explored freer techniques, pioneered by Loie Fuller,
Isadora Duncan, Mary Wigman, and Ruth St. Denis. Émile Jaques-Dalcroze’s Eurhythmics linked
</item 10>
=== END ITEM id=2fe1fdc02471aa93 ===

=== ITEM id=03017b5ebeb281c3 ===
Gold sentences and facts:
Sentence 0: Bay of Montevideo
  Fact 0.0: Bay of Montevideo

Retrieved items:
<item 1>
FIFA World Cup hosts
Under FIFA rules as of 2017 , the 2030 World Cup can not be held in Asia ( AFC ) because the Asian Football Confederation is excluded from the bidding following
the selection of Qatar in 2022 . Also in June 2017 , UEFA 's president Aleksander Čeferin stated that Europe ( UEFA ) will definitely fight for its right to
host the 2030 World Cup . The Uruguay -- Argentina bid would coincide with the centennial anniversary of the first FIFA World Cup final , and the bicentennial
of the first Constitution of Uruguay . The Uruguay - Argentina bid was officially confirmed on 29 July 2017 . Paraguay is aiming for joining the bid , but this
is not confirmed or discarded yet . The efforts for the inclusion of Paraguay in the bid caused discomfort in the Uruguayan Football Association , and Wilmar
Valdéz ( AUF president ) accused some insistence of the Conmebol although it is not official that Paraguay joins the event . Chile also wants to organize the
World Cup with Uruguay and Argentina . A joint bid was announced by the Argentine Football Association and the Uruguayan Football Association on 29 July 2017 .
</item 1>

<item 2>
Dead Sea
Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran Chaco Chiquitano dry forests Valdes Peninsula West Andes Tropical Andes Wet Andes
Dry Andes Pariacaca mountain range Altiplano Atacama Desert Latin Hispanic American Cordillera Ring of Fire LAC Polar regions Antarctic Antarctic Peninsula East
Antarctica West Antarctica Eklund Islands Ecozone Extreme points Islands Arctic Arctic Alaska British Arctic Territories Canadian Arctic Archipelago Finnmark
Greenland Northern Canada Northwest Territories Nunavik Nunavut Russian Arctic Sakha Sápmi Yukon North American Arctic Earth 's oceans and seas Arctic Ocean
Amundsen Gulf Barents Sea Beaufort Sea Chukchi Sea East Siberian Sea Greenland Sea Gulf of Boothia Kara Sea Laptev Sea Lincoln Sea Prince Gustav Adolf Sea
Pechora Sea Queen Victoria Sea Wandel Sea White Sea Atlantic Ocean Adriatic Sea Aegean Sea Alboran Sea Archipelago Sea Argentine Sea Baffin Bay Balearic Sea
Baltic Sea Bay of Biscay Bay of Bothnia Bay of Campeche Bay of Fundy Black Sea Bothnian Sea Caribbean Sea Celtic Sea English Channel Foxe Basin Greenland Sea
Gulf of Bothnia Gulf of
</item 2>

<item 3>
Geography of Nigeria
Plains rise to the north of the valleys . To the southwest of the Niger there is `` rugged '' highland , and to the southeast of the Benue hills and mountains
are found all the way to the border with Cameroon . Coastal plains are found in both the southwest and the southeast . Niger Delta ( edit ) This section needs
expansion . You can help by adding to it . ( January 2011 ) The Niger Delta is located in the southern part of Nigeria . It is one of the world 's largest
arcuate fan - shaped river deltas . The riverine area of the Niger Delta is a coastal belt of swamps bordering the Atlantic ocean . The swamps are vegetated
tidal flats formed by a reticulate pattern of interconnected meandering creeks and distributaries of the River Niger . Vegetation of the Niger Delta ( edit )
The vegetation of the Niger Delta consist mainly of forest swamps . The forest are of two types , nearest the sea is a belt of saline / brackish Mangrove swamp
separated from the sea by sand beach ridges within the mangrove swamp . Numerous sandy islands occur with fresh water vegetation .
</item 3>

<item 4>
FIFA World Cup hosts
Retrieved 14 January 2017 . Jump up ^ `` UEFA will ' fight ' for Europe 's right to host 2030 World Cup '' . Retrieved 2 June 2017 . ^ Jump up to : Molestia en
Uruguay por video de Conmebol ( in Spanish ) Jump up ^ ElPais . `` Chile no se rinde con ser sede '' ( in Spanish ) . Retrieved 15 March 2018 . Jump up ^ Dudko
, James ( 29 July 2017 ) . `` Argentina , Uruguay Announce Bid to Host 2030 FIFA World Cup '' . Bleacher Report . Turner Broadcasting System . Jump up ^ Gadd ,
Mick ( 1 September 2017 ) . `` Lionel Messi and Luis Suarez campaign for Uruguay and Argentina 's 2030 World Cup bid before 0 - 0 draw '' . Daily Mirror . Jump
up ^ `` Es oficial : Paraguay se suma a la candidatura de Argentina y Uruguay para el Mundial 2030 '' ( It 's official : Paraguay adds itself to the Argentina
and Uruguay bid ) ( in Spanish ) .
</item 4>

<item 5>
Category:Bodies of water of Uruguay
Category : Bodies of water of Uruguay - wikipedia Help Category : Bodies of water of Uruguay Jump to : navigation , search Wikimedia Commons has media related
to Bodies of water in Uruguay . This category has the following 4 subcategories , out of 4 total . ► Bays of Uruguay ‎ ( 1 P ) ► Lakes of Uruguay ‎ ( 5 P ) ►
Rivers of Uruguay ‎ ( 2 C , 35 P ) ► Springs of Uruguay ‎ ( 1 P ) Pages in category `` Bodies of water of Uruguay '' This category contains only the following
page . This list may not reflect recent changes ( learn more ) . Bay of Montevideo Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Category:Bodies_of_water_of_Uruguay&oldid=756422847 '' Categories : Landforms of Uruguay Bodies of water by country
Water in Uruguay Bodies of water of South America by country Talk Help About Wikipedia Azərbaycanca Deutsch فارسی 한국어 Magyar Português Türkçe اردو Edit links
This page was last edited on 24 December 2016 , at 03 : 52 . About Wikipedia
</item 5>

<item 6>
Pan-American Highway
In Chile , the highway follows Chile Route 5 south to a point north of Santiago ( Llaillay ) , where the highway splits into two parts , one of which goes
through Chilean territory to Puerto Montt , where it splits again , to Quellón on Chiloé Island , and to its continuation as the Carretera Austral . The other
part goes east along Chile Route 60 , which becomes Argentina National Route 7 at the Argentinian border and continues to Buenos Aires , the end of the main
highway . The highway network also continues south of Buenos Aires along Argentina National Route 3 towards the city of Ushuaia in Tierra del Fuego . Another
branch , from Buenos Aires to Asunción in Paraguay , heads out of Buenos Aires on Argentina National Route 9 . It switches to Argentina National Route 11 at
Rosario , which crosses the border with Paraguay right at Asunción . Other branches probably exist across the center of South America .
</item 6>

<item 7>
Dead Sea
Basin High Desert Monterey Peninsula Upper Peninsula of Michigan Lower Peninsula of Michigan Virginia Peninsula Keweenaw Peninsula Middle Peninsula Delmarva
Peninsula Alaska Peninsula Kenai Peninsula Niagara Peninsula Beringia Belt regions Bible Belt Black Belt Corn Belt Cotton Belt Frost Belt Rice Belt Rust Belt
Sun Belt Snow Belt Latin Northern Mexico Baja California Peninsula Gulf of California Colorado River Delta Gulf of Mexico Soconusco Tierra Caliente La Mixteca
La Huasteca Bajío Valley of Mexico Mezquital Valley Sierra Madre de Oaxaca Yucatán Peninsula Basin and Range Province Western Caribbean Zone Isthmus of Panama
Gulf of Panama Pearl Islands Azuero Peninsula Mosquito Coast West Indies Antilles Greater Antilles Lesser Antilles Leeward Leeward Antilles Windward Lucayan
Archipelago Southern Caribbean Aridoamerica Mesoamerica Oasisamerica Northern Middle Anglo Latin French Hispanic American Cordillera Ring of Fire LAC Regions of
Oceania Australasia Gulf of Carpentaria New Guinea Bonis Peninsula Papuan Peninsula Huon Peninsula Huon Gulf Bird 's Head Peninsula Gazelle Peninsula New
Zealand South Island North Island Coromandel Peninsula Zealandia New Caledonia Solomon Islands ( archipelago ) Vanuatu Kula Gulf Australia Capital Country
</item 7>

<item 8>
FIFA World Cup hosts
The original bid logo for the Uruguay - Argentina World Cup Bid . Confirmed 2030 bids : CONMEBOL : Uruguay / Argentina / Paraguay : Uruguay -- Argentina --
Paraguay 2030 FIFA World Cup bid Expressed interest in bidding : UEFA : England : England 2030 FIFA World Cup bid 2034 FIFA World Cup ( edit ) Main article :
2034 FIFA World Cup The first bid for the 2034 FIFA World Cup has been proposed as a collective bid by the members of the Association of Southeast Asian Nations
. The idea of a combined ASEAN bid had been mooted as early as January 2011 , when the Football Association of Singapore 's then - President , Zainudin Nordin ,
said in a statement that the proposal had been made at an ASEAN Foreign Ministers meeting . In 2013 , Nordin and Special Olympics Malaysia President , Datuk
Mohamed Feisol Hassan , recalled the idea for ASEAN to jointly host a World Cup . Under FIFA rules as of 2017 , the 2030 World Cup can not be held in Asia ( AFC
) as Asian Football Confederation members are excluded from the bidding following the selection of Qatar in 2022 .
</item 8>

<item 9>
Pan-American Highway
Other branches probably exist across the center of South America . Almost all Pan-American sections in Gran Buenos Aires are modern and fast main highways West
Access to Buenos Aires Brazil and Uruguay ( edit ) A continuation of the Pan-American Highway to the Brazilian cities of São Paulo and Rio de Janeiro uses a
ferry from Buenos Aires to Colonia in Uruguay and Uruguay Highway 1 to Montevideo . Uruguay Highway 9 and Brazil Highway 471 route to near Pelotas , from where
Brazil Highway 116 leads to Brazilian main cities . Guyana , Suriname and French Guyana ( edit ) The highway does not have official segments to Belize , Guyana
, Suriname and French Guiana , nor to any of the island nations in the Americas . However , highways from Venezuela link to Brazilian Trans - Amazonian highway
that provides a southwest entrance to Guyana , route to the coast , and follow a coastal route through Suriname to French Guiana . West Indies section ( edit )
Plans have been discussed for including the West Indies in the Pan American Highway system . According to these , a system of ferries would be established to
connect terminal points of the highway .
</item 9>

<item 10>
Dead Sea
New Caledonia Solomon Islands ( archipelago ) Vanuatu Kula Gulf Australia Capital Country Eastern Australia Lake Eyre basin Murray -- Darling basin Northern
Australia Nullarbor Plain Outback Southern Australia Maralinga Sunraysia Great Victoria Desert Gulf of Carpentaria Gulf St Vincent Lefevre Peninsula Fleurieu
Peninsula Yorke Peninsula Eyre Peninsula Mornington Peninsula Bellarine Peninsula Mount Henry Peninsula Melanesia Islands Region Bismarck Archipelago Solomon
Islands Archipelago Fiji New Caledonia Papua New Guinea Vanuatu Micronesia Caroline Islands Federated States of Micronesia Palau Guam Kiribati Marshall Islands
Nauru Northern Mariana Islands Wake Island Polynesia Easter Island Hawaiian Islands Cook Islands French Polynesia Austral Islands Gambier Islands Marquesas
Islands Society Islands Tuamotu Kermadec Islands Mangareva Islands Samoa Tokelau Tonga Tuvalu Ring of Fire Regions of South America East Amazon basin Atlantic
Forest Caatinga Cerrado North Caribbean South America West Indies Los Llanos The Guianas Amazon basin Amazon rainforest Gulf of Paria Paria Peninsula Paraguaná
Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran
</item 10>
=== END ITEM id=03017b5ebeb281c3 ===

=== ITEM id=b87d69329a44fc5a ===
Gold sentences and facts:
Sentence 0: The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at Newport Ranch , a resort and cattle ranch to the north of San Francisco . It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on May 4 , 2017 , and shows Brown and Alaina cruising down a road .
  Fact 0.0: The official music video for the song was directed by P.R. Brown.
  Fact 0.1: The official music video for the song was produced by Steve Lamar.
  Fact 0.2: The video was filmed along the Californian coast at The Inn at Newport Ranch.
  Fact 0.3: The Inn at Newport Ranch is a resort and cattle ranch.
  Fact 0.4: The Inn at Newport Ranch is north of San Francisco.
  Fact 0.5: The video features Brown and his duet partner Lauren Alaina singing along the coast.
  Fact 0.6: The video was released on May 14, 2017.
  Fact 0.7: An earlier lyric video was first released on May 4, 2017.
  Fact 0.8: The earlier lyric video shows Brown and Alaina cruising down a road.

Retrieved items:
<item 1>
What Ifs
Jump up ^ Bjorke , Matt ( January 3 , 2018 ) . `` Top 30 Country Digital Singles Chart : January 3 , 2018 '' . Roughstock . Retrieved January 14 , 2017 . ^ Jump
up to : Bjorke , Matt ( March 6 , 2018 ) . `` The Top 30 Digital Country Singles : March 5 , 2018 '' . Roughstock . Retrieved March 11 , 2018 . Jump up ^ ``
Watch Kane Brown , Lauren Alaina 's Picturesque ' What Ifs ' Video '' . Rolling Stone . May 15 , 2017 . Jump up ^ Kimble , Lindsay ( May 4 , 2017 ) . `` From
Middle School Chorus to Nashville , Kane Brown and Lauren Alaina Team Up for Duet ' What Ifs ' '' . Jump up ^ `` Kane Brown Chart History ( Canadian Hot 100 )
'' . Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard .
</item 1>

<item 2>
What Ifs
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
</item 2>

<item 3>
Graph query row:
video: official music video for What Ifs
video_relationship: LOCATED_IN
filming_location: The Inn at Newport Ranch
coast_relationship: LOCATED_IN
coast: Californian coast
</item 3>

<item 4>
What Ifs
Taste of Country . ^ Jump up to : Konicki , Lisa ( February 8 , 2017 ) . `` Kane Brown 's Current Single , `` What Ifs , '' With Lauren Alaina Almost Did n't
Make The Album `` . Nash Country Daily . Jump up ^ Freeman , Jon ( December 2 , 2016 ) . `` Kane Brown on Personal New Album , Why He Wo n't Sing First Hit :
Ram Report '' . Rolling Stone . ^ Jump up to : Asker , Jim ( October 17 , 2017 ) . `` Kane Brown Becomes First Artist To Simultaneously Lead Five Country Charts
'' . Billboard . Jump up ^ `` ' What Ifs ' sheet music '' . MusicNotes.com . Retrieved 5 February 2018 . Jump up ^ Bjorke , Matt ( December 13 , 2016 ) . `` Top
30 Digital Country Singles Chart : December 13 , 2016 '' . Roughstock . Retrieved February 5 , 2017 . Jump up ^ `` Country Airplay '' . Billboard .
</item 4>

<item 5>
What Ifs
wikipedia.org/w/index.php?title=What_Ifs&oldid=838232791 '' Categories : 2016 songs 2017 singles Kane Brown songs Lauren Alaina songs RCA Records Nashville
singles Vocal duets Song recordings produced by Dann Huff Songs written by Jordan Schmidt Billboard Hot Country Songs number - one singles Hidden categories :
Music infoboxes with deprecated parameters Articles with hAudio microformats Singlechart usages for Canada Singlechart called without song Singlechart usages
for Billboardcanadacountry Singlechart usages for Billboardhot100 Singlechart usages for Billboardcountryairplay Singlechart usages for Billboardcountrysongs
Certification Table Entry usages for Canada Certification Table Entry usages for United States Talk Contents About Wikipedia Español Edit links This page was
last edited on 25 April 2018 , at 18 : 59 . About Wikipedia
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
What Ifs
What Ifs - wikipedia What Ifs Jump to : navigation , search `` What Ifs '' Single by Kane Brown featuring Lauren Alaina from the album Kane Brown Released
February 6 , 2017 ( 2017 - 02 - 06 ) Format Digital download Genre Country Length 3 : 08 Label RCA Nashville Songwriter ( s ) Kane Brown Matthew McGinn Jordan
Schmidt Producer ( s ) Dann Huff Kane Brown singles chronology `` Thunder in the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) `` Thunder in
the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) Lauren Alaina singles chronology `` Road Less Traveled '' ( 2016 ) Road Less Traveled 2016 ``
What Ifs '' ( 2017 ) What Ifs2017 `` Doin ' Fine '' ( 2017 ) Doin ' Fine 2017 `` What Ifs '' is a song recorded by American country music singer Kane Brown
featuring Lauren Alaina for his self - titled debut album .
</item 7>

<item 8>
What Ifs
The song is the first No. 1 on the chart for both Brown and Alaina . The following week , for charts dated October 28 , 2017 , `` What Ifs '' stayed at number
one , as well as topping the Country Airplay and Country Streaming Songs charts . Together with the release of the deluxe edition of the album which also
reached number on Top Country Albums , and another number one for a new track `` Heaven '' on digital sales , Brown became the first artist to have simultaneous
number ones on all five main country charts . The song was certified Platinum by the RIAA on October 2 , 2017 . It was the third most - consumed country song of
2017 , with 1,551,000 including streams ( 599,000 in pure sale ) sold in the year . It has sold 679,000 copies in the US as of March 2018 . Music video ( edit )
The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at
Newport Ranch , a resort and cattle ranch to the north of San Francisco .
</item 8>

<item 9>
What Ifs
The song was released with the album through RCA Nashville and was serviced to radio as the third single on February 6 , 2017 . Contents ( hide ) 1 Background 2
Composition 3 Commercial performance 4 Music video 5 Chart performance 5.1 Weekly charts 5.2 Year - end charts 6 Certifications and sales 7 References
Background ( edit ) The song was written by Brown with Matthew McGinn and Jordan Schmidt . The song is about a couple where the question `` what if '' is asked
repeatedly about what would happen in their relationship . It features a duet with Lauren Alaina , a school friend from Lakeview - Fort Oglethorpe High School
when he was living in Fort Oglethorpe , Georgia . According to Brown , the song was initially not intended to be a duet , he said : `` Once we decided on adding
a woman to `` What Ifs , '' I knew that I wanted Lauren to sing it , and , of course , she killed it . ''
</item 9>

<item 10>
What Ifs
According to Brown , another song was going to be used in the album , however , he was asked so much about `` What Ifs '' on social media that it was then
decided that the song should be included in the album . Composition ( edit ) The song is in the key of G minor with a moderately fast tempo of approximately 126
beats per minute . It follows the chord progression Gm - B ♭ - F. Brown 's vocals range from F - B ♭ . Commercial performance ( edit ) The song sold 6,000
downloads in the US the week the album was released on December 2 , 2016 , allowing it to chart at number 37 on the US Billboard Hot Country Songs chart . The
song first entered the US Billboard Country Airplay chart for the week of March 4 , 2017 at number 60 . It reached No. 1 on Hot Country Songs in October 2017 ,
ending the record breaking 34 - week reign of Sam Hunt 's `` Body Like a Back Road '' . The song is the first No. 1 on the chart for both Brown and Alaina .
</item 10>
=== END ITEM id=b87d69329a44fc5a ===

=== ITEM id=818663a1f2600f0f ===
Gold sentences and facts:
Sentence 0: An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine .
  Fact 0.0: An elevator with a counterbalance approximates an ideal Atwood machine.
  Fact 0.1: A counterbalanced elevator relieves the driving motor from the load of holding the elevator cab.
  Fact 0.2: A counterbalanced elevator's driving motor has to overcome only the weight difference and inertia of the two masses.
  Fact 0.3: The same principle is used for funicular railways with two connected railway cars on inclined tracks.
  Fact 0.4: The same principle is used for the elevators on the Eiffel Tower, which counterbalance each other.
  Fact 0.5: Ski lifts are another example.
  Fact 0.6: In ski lifts, the gondolas move on a closed (continuous) pulley system up and down the mountain.
  Fact 0.7: A ski lift is similar to the counter-weighted elevator, but with a constraining force provided by the cable in the vertical dimension.
  Fact 0.8: A ski lift achieves work in both the horizontal and vertical dimensions.
  Fact 0.9: Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine.

Retrieved items:
<item 1>
Atwood machine
Atwood machine - wikipedia Atwood machine Jump to : navigation , search Illustration of Atwood machine , 1905 . The Atwood machine ( or Atwood 's machine ) was
invented in 1784 by the English mathematician George Atwood as a laboratory experiment to verify the mechanical laws of motion with constant acceleration .
Atwood 's machine is a common classroom demonstration used to illustrate principles of classical mechanics . The ideal Atwood Machine consists of two objects of
mass m and m , connected by an inextensible massless string over an ideal massless pulley . When m = m , the machine is in neutral equilibrium regardless of the
position of the weights . When m ≠ m both masses experience uniform acceleration . Contents ( hide ) 1 Equation for constant acceleration 2 Equation for tension
3 Equations for a pulley with inertia and friction 3.1 Practical implementations 4 See also 5 Notes 6 External links Equation for constant acceleration ( edit )
The free body diagrams of the two hanging masses of the Atwood machine .
</item 1>

<item 2>
Atwood machine
ISBN 81 - 85015 - 53 - 8 . Section 1 - 6 , example 2 Jump up ^ See Wikipedia article : https://en.wikipedia.org/wiki/Boat_lift External links ( edit ) Professor
Greenslade 's account on the Atwood Machine Atwood 's Machine by Enrique Zeleny , The Wolfram Demonstrations Project . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Atwood_machine&oldid=792136529 '' Categories : Mechanics Physics experiments Talk Contents About Wikipedia Català
Deutsch Español Français 한국어 Hrvatski Italiano Nederlands 日本 語 Polski Português Русский Srpskohrvatski / српскохрватски Türkçe Українська 中文 Edit links This
page was last edited on 24 July 2017 , at 17 : 58 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy .
</item 2>

<item 3>
Atwood machine
Our sign convention , depicted by the acceleration vectors is that m accelerates downward and that m accelerates upward , as would be the case if m > m We are
able to derive an equation for the acceleration by using force analysis . If we consider a massless , inextensible string and an ideal massless pulley , the
only forces we have to consider are : tension force ( T ) , and the weight of the two masses ( W and W ) . To find an acceleration we need to consider the
forces affecting each individual mass . Using Newton 's second law ( with a sign convention of m 1 > m 2 ( \ displaystyle m_ ( 1 ) > m_ ( 2 ) ) ) we can derive
a system of equations for the acceleration ( a ) . As a sign convention , we assume that a is positive when downward for m 1 ( \ displaystyle m_ ( 1 ) ) and
upward for m 2 ( \ displaystyle m_ ( 2 ) ) .
</item 3>

<item 4>
Atwood machine
Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine . See also ( edit ) Wikimedia Commons has media related to
Atwood 's machine . Frictionless plane Kater 's pendulum Spherical cow Swinging Atwood 's machine Notes ( edit ) Jump up ^ Tipler , Paul A. ( 1991 ) . Physics
For Scientists and Engineers ( 3rd , extended ed . ) . New York : Worth Publishers . p. 160 . ISBN 0 - 87901 - 432 - 6 . Chapter 6 , example 6 - 13 Jump up ^
Goldstein , Herbert ( 1980 ) . Classical Mechanics ( 2nd ed . ) . New Delhi : Addison - Wesley / Narosa Indian Student Edition . pp. 26 -- 27 . ISBN 81 - 85015
- 53 - 8 .
</item 4>

<item 5>
Atwood machine
+ τ f r i c t i o n r g ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) + ( ( \ tau _ ( \ mathrm (
friction ) ) ) \ over ( rg ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Should bearing friction be negligible ( but not the
inertia of the pulley and not the traction of the string on the pulley rim ) , these equations simplify as the following results : Acceleration : a = g ( m 1 −
m 2 ) m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) )
</item 5>

<item 6>
Atwood machine
a = g m 1 − m 2 m 1 + m 2 ( \ displaystyle a = g ( m_ ( 1 ) - m_ ( 2 ) \ over m_ ( 1 ) + m_ ( 2 ) ) ) For example , substituting into m 1 a = m 1 g − T ( \
displaystyle m_ ( 1 ) a = m_ ( 1 ) g-T ) , results in T = 2 g m 1 m 2 m 1 + m 2 = 2 g 1 / m 1 + 1 / m 2 ( \ displaystyle T = ( 2gm_ ( 1 ) m_ ( 2 ) \ over m_ ( 1
) + m_ ( 2 ) ) = ( 2g \ over 1 / m_ ( 1 ) + 1 / m_ ( 2 ) ) ) Equations for a pulley with inertia and friction ( edit ) For very small mass differences between m
and m , the rotational inertia I of the pulley of radius r can not be neglected .
</item 6>

<item 7>
Atwood machine
and our concluding formula for acceleration a = g m 1 − m 2 m 1 + m 2 ( \ displaystyle a = g ( m_ ( 1 ) - m_ ( 2 ) \ over m_ ( 1 ) + m_ ( 2 ) ) ) Conversely ,
the acceleration due to gravity , g , can be found by timing the movement of the weights , and calculating a value for the uniform acceleration a : a = 2 d t 2
( \ displaystyle a = ( \ frac ( 2d ) ( t ^ ( 2 ) ) ) ) . The Atwood machine is sometimes used to illustrate the Lagrangian method of deriving equations of
motion . Equation for tension ( edit ) It can be useful to know an equation for the tension in the string . To evaluate tension , substitute the equation for
acceleration in either of the 2 force equations .
</item 7>

<item 8>
Atwood machine
( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Practical implementations ( edit ) Atwood 's original illustrations show the main pulley 's axle
resting on the rims of another four wheels , to minimize friction forces from the bearings . Many historical implementations of the machine follow this design .
An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it
has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on
inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a
closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided
by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted
elevator system approximating an Atwood machine .
</item 8>

<item 9>
Atwood machine
The angular acceleration of the pulley is given by the no - slip condition : α = a r , ( \ displaystyle \ alpha = ( a \ over r ) , ) where α ( \ displaystyle \
alpha ) is the angular acceleration . The net torque is then : τ n e t = ( T 1 − T 2 ) r − τ f r i c t i o n = I α ( \ displaystyle \ tau _ ( \ mathrm ( net ) )
= \ left ( T_ ( 1 ) - T_ ( 2 ) \ right ) r - \ tau _ ( \ mathrm ( friction ) ) = I \ alpha ) Combining with Newton 's second law for the hanging masses , and
solving for T , T , and a , we get : Acceleration : a = g ( m 1 − m 2 ) − τ f r i c t i o n r m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 )
) - ( \ tau _ ( \ mathrm ( friction ) ) \ over r ) ) \ over ( m_ (
</item 9>

<item 10>
Atwood machine
Weight of m 1 ( \ displaystyle m_ ( 1 ) ) and m 2 ( \ displaystyle m_ ( 2 ) ) is simply W 1 = m 1 g ( \ displaystyle W_ ( 1 ) = m_ ( 1 ) g ) and W 2 = m 2 g ( \
displaystyle W_ ( 2 ) = m_ ( 2 ) g ) respectively . Forces affecting m : m 1 g − T = m 1 a ( \ displaystyle \ ; m_ ( 1 ) g-T = m_ ( 1 ) a ) Forces affecting m :
T − m 2 g = m 2 a ( \ displaystyle \ ; T - m_ ( 2 ) g = m_ ( 2 ) a ) and adding the two previous equations we obtain m 1 g − m 2 g = m 1 a + m 2 a ( \
displaystyle \ ; m_ ( 1 ) g - m_ ( 2 ) g = m_ ( 1 ) a + m_ ( 2 ) a ) ,
</item 10>
=== END ITEM id=818663a1f2600f0f ===

=== ITEM id=824aa0a6e67211cf ===
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
Humus has many nutrients that improve the health of soil , nitrogen being the most important . The ratio of carbon to nitrogen ( C : N ) of humus is 10 : 1 .
Contents ( hide ) 1 Description 2 Humification 3 Stability 4 Horizons 5 Benefits of soil organic matter and humus 6 See also 7 References 8 External links
Description ( edit ) It is difficult to define humus precisely because it is a very complex substance which is not fully understood . Humus is different from
decomposing soil organic matter . The latter looks rough and has visible remains of the original plant or animal matter . Fully humified humus , on the contrary
, has a uniformly dark , spongy , and jelly - like appearance , and is amorphous ; it may gradually decompose over several years or persist for millennia . It
has no determinate shape , structure , or quality . However , when examined under a microscope , humus may reveal tiny plant , animal , or microbial remains
that have been mechanically , but not chemically , degraded . This suggests an ambiguous boundary between humus and soil organic matter .
</item 1>

<item 2>
Humus
This suggests an ambiguous boundary between humus and soil organic matter . While distinct , humus is an integral part of soil organic matter . Humification (
edit ) Microorganisms decompose a large portion of the soil organic matter into inorganic minerals that the roots of plants can absorb as nutrients . This
process is termed `` mineralization '' . In this process , nitrogen ( nitrogen cycle ) and the other nutrients ( nutrient cycle ) in the decomposed organic
matter are recycled . Depending on the conditions in which the decomposition occurs , a fraction of the organic matter does not mineralize , and instead is
transformed by a process called `` humification '' into concatenations of organic polymers . Because these organic polymers are resistant to the action of
microorganisms , they are stable , and constitute humus . This stability implies that humus integrates into the permanent structure of the soil , thereby
improving it . Humification can occur naturally in soil or artificially in the production of compost . Organic matter is humified by a combination of
saprotrophic fungi , bacteria , microbes and animals such as earthworms , nematodes , protozoa , and arthropods .
</item 2>

<item 3>
Humus
Soil Biology and Biochemistry . 26 ( 2 ) : 183 -- 220 . doi : 10.1016 / 0038 - 0717 ( 94 ) 90161 - 9 . Jump up ^ Humintech ® Definition Of Soil Organic Matter &
Humic Acids Based Products Jump up ^ Brady , N.C. ; Weil , R.R. ( 1999 ) . The nature and properties of soils . Upper Saddle River , N.J. : Prentice Hall , Inc
. Jump up ^ Soil biology Jump up ^ Berg , B. , McClaugherty , C. , 2007 . Plant litter : decomposition , humus formation , carbon sequestration , 2nd ed .
Springer , 338 pp. , ISBN 3 - 540 - 74922 - 5 Jump up ^ Levin , L. ; Forchiassin , F. ; Ramos , A.M. ( 2002 ) . `` Copper induction of lignin - modifying
enzymes in the white - rot fungus Trametes trogii '' . Mycologia. 94 : 377 -- 383 .
</item 3>

<item 4>
Humus
Most soils have three major horizons : the surface horizon ( A ) , the subsoil ( B ) , and the substratum ( C ) . Some soils have an organic horizon ( O ) on
the surface , but this horizon can also be buried . The master horizon ( E ) is used for subsurface horizons that have significantly lost minerals ( eluviation
) . Bedrock , which is not soil , uses the letter R . Benefits of soil organic matter and humus ( edit ) The importance of chemically stable humus is thought by
some to be the fertility it provides to soils in both a physical and chemical sense , though some agricultural experts put a greater focus on other features of
it , such as its ability to suppress disease . It helps the soil retain moisture by increasing microporosity , and encourages the formation of good soil
structure . The incorporation of oxygen into large organic molecular assemblages generates many active , negatively charged sites that bind to positively
charged ions ( cations ) of plant nutrients , making them more available to the plant by way of ion exchange . Humus allows soil organisms to feed and reproduce
, and is often described as the `` life - force '' of the soil .
</item 4>

<item 5>
Humus
, 2011 . Accessed 24 Nov 2011 . < http://www.britannica.com/EBchecked/topic/276408/humus > . Jump up ^ Chertov , O.G. ; Kornarov , A.S. ; Crocker , G. ; Grace ,
P. ; Klir , J. ; Körschens , M. ; Poulton , P.R. ; Richter , D. ( 1997 ) . `` Simulating trends of soil organic carbon in seven long - term experiments using
the SOMM model of the humus types '' . Geoderma. 81 : 121 -- 135 . doi : 10.1016 / S0016 - 7061 ( 97 ) 00085 - 2 . Jump up ^ Baritz , R. , 2003 . Humus forms in
forests of the northern German lowlands . Schweizerbart , Stuttgart , Germany , 145 pp. ( 2 ) Jump up ^ Bunting , B.T. ; Lundberg , J. ( 1995 ) .
</item 5>

<item 6>
Humus
Humus is a colloidal substance and increases the cation exchange capacity of soil , hence its ability to store nutrients by chelation . While these nutrient
cations are available to plants , they are held in the soil and prevented from being leached by rain or irrigation . Humus can hold the equivalent of 80 -- 90 %
of its weight in moisture , and therefore increases the soil 's capacity to withstand drought . The biochemical structure of humus enables it to moderate , i.e.
buffer , excessive acidic or alkaline soil conditions . During humification , microbes secrete sticky , gum - like mucilages ; these contribute to the crumby
structure ( tilth ) of the soil by adhering particles together and allowing greater aeration of the soil . Toxic substances such as heavy metals and excess
nutrients can be chelated , i.e. , bound to the organic molecules of humus , and so prevented from leaching away . The dark , usually brown or black , color of
humus helps to warm cold soils in Spring .
</item 6>

<item 7>
Humus
Humus has a characteristic black or dark brown color and is an accumulation of organic carbon . Besides the three major soil horizons of ( A ) surface / topsoil
, ( B ) subsoil , and ( C ) substratum , some soils have an organic horizon ( O ) on the very surface . Hard bedrock ( R ) is not in a strict sense soil . In
soil science , humus ( derived in 1790 -- 1800 from the Latin humus for earth , ground ) denominates the fraction of soil organic matter that is amorphous and
without the `` cellular cake structure characteristic of plants , micro-organisms or animals . '' Humus significantly affects the bulk density of soil and
contributes to its retention of moisture and nutrients . In agriculture , `` humus '' sometimes also is used to describe mature or natural compost extracted
from a woodland or other spontaneous source for use as a soil conditioner . It is also used to describe a topsoil horizon that contains organic matter ( humus
type , humus form , humus profile ) . Humus is the dark organic matter that forms in soil when dead plant and animal matter decays .
</item 7>

<item 8>
Humus
Stability ( edit ) Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as
stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of
soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as
less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable
form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated
to have been important in the formation of the very fertile Amazonian terra preta do Indio . Horizons ( edit ) Humus has a characteristic black or dark brown
color and is organic due to an accumulation of organic carbon . Soil scientists use the capital letters O , A , B , C , and E to identify the master horizons ,
and lowercase letters for distinctions of these horizons .
</item 8>

<item 9>
Humus
The process that converts soil organic matter into humus feeds the population of microorganisms and other creatures in the soil , and thus maintains high and
healthy levels of soil life . The rate at which soil organic matter is converted into humus promotes ( when fast ) or limits ( when slow ) the coexistence of
plants , animals , and microorganisms in the soil . Effective humus and stable humus are additional sources of nutrients for microbes : the former provides a
readily available supply and the latter acts as a longeval storage reservoir . Decomposition of dead plant material causes complex organic compounds to be
slowly oxidized ( lignin - like humus ) or to decompose into simpler forms ( sugars and amino sugars , and aliphatic and phenolic organic acids ) , which are
further transformed into microbial biomass ( microbial humus ) or reorganized , and further oxidized , into humic assemblages ( fulvic and humic acids ) , which
bind to clay minerals and metal hydroxides . The ability of plants to absorb humic substances with their roots and metabolize them has been long debated . There
is now a consensus that humus functions hormonally rather than simply nutritionally in plant physiology .
</item 9>

<item 10>
Humus
Plant remains , including those that animals digested and excreted , contain organic compounds : sugars , starches , proteins , carbohydrates , lignins , waxes
, resins , and organic acids . Decay in the soil begins with the decomposition of sugars and starches from carbohydrates , which decompose easily as
detritivores initially invade the dead plant organs , while the remaining cellulose and lignin decompose more slowly . Simple proteins , organic acids ,
starches , and sugars decompose rapidly , while crude proteins , fats , waxes , and resins remain relatively unchanged for longer periods of time . Lignin ,
which is quickly transformed by white - rot fungi , is one of the primary precursors of humus , together with by - products of microbial and animal activity .
The humus produced by humification is thus a mixture of compounds and complex biological chemicals of plant , animal , or microbial origin that has many
functions and benefits in soil . Some judge earthworm humus ( vermicompost ) to be the optimal organic manure .
</item 10>
=== END ITEM id=824aa0a6e67211cf ===
