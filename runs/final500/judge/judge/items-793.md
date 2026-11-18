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

=== ITEM id=f51977de1cedfa1d ===
Gold sentences and facts:
Sentence 0: The Pan-American Highway is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) , called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World Records , the Pan-American Highway is the world 's longest `` motorable road '' . However , because of the Darién Gap , it is not possible to cross between South America and Central America with conventional highway vehicles . Without an all - terrain vehicle , it is necessary to circumnavigate this terrestrial stretch by sea .
  Fact 0.0: The Pan-American Highway is a network of roads.
  Fact 0.1: The Pan-American Highway measures about 30,000 kilometres (19,000 mi) in total length.
  Fact 0.2: The Pan-American Highway has a rainforest break of approximately 160 km (100 mi).
  Fact 0.3: The rainforest break of the Pan-American Highway is called the Darién Gap.
  Fact 0.4: Apart from the Darién Gap, the Pan-American Highway links almost all of the Pacific coastal countries of the Americas.
  Fact 0.5: The Pan-American Highway links those countries in a connected highway system.
  Fact 0.6: According to Guinness World Records, the Pan-American Highway is the world's longest 'motorable road'.
  Fact 0.7: Because of the Darién Gap, it is not possible to cross between South America and Central America with conventional highway vehicles.
  Fact 0.8: Without an all-terrain vehicle, it is necessary to go around the terrestrial stretch of the Darién Gap by sea.

Retrieved items:
<item 1>
Graph query row:
highway: Pan-American Highway
relationship: RELATED_TO
organization: Guinness World Records
fact: None
context: None
relationship_detail: None
</item 1>

<item 2>
Pan-American Highway
Travelers would then be able to ferry from Key West to Havana , drive to the eastern tip of Cuba , ferry to Haiti , drive through Haiti and the Dominican
Republic , and ferry again to Puerto Rico . Included in this system would also be a ferry from the western tip of Cuba to the Yucatán Peninsula . Mexico has
already surveyed a route which will run across the Yucatán , Campeche , and Chiapas to San Cristobal de Las Casas , on the Pan American Highway . ( `` The Pan
American Highway System '' by Travel Division Pan American Union , Washington D.C. October 1947 ) Art and culture ( edit ) Travel writer Tim Cahill wrote a book
, Road Fever , about his record - setting 24 - day drive from Ushuaia in the Argentine province of Tierra del Fuego to Prudhoe Bay in the U.S. state of Alaska
with professional long - distance driver Garry Sowerby , much of their route following the Pan-American Highway . In the British motoring show Top Gear , the
presenters drove on a section of the road in their off - road vehicles in the Bolivian Special .
</item 2>

<item 3>
Pan-American Highway
Panamericana -- Pan American Highway -- in Pichincha , Ecuador , near Cashapamba Panamericana -- northern Peru near Pacasmayo Panamericana near Puerto De Lomas
, Peru Panamericana in the Atacama Desert northern Chile GMC Suburban traveled all of the Pan-American Highway . Patagonia , Chile . Panamericana - Pan American
Highway - in the Cordillera de los Andes , southern Ecuador , near to Catacocha , 2500m / 8000 ft Elev . National Route 7 -- Pan-American Highway in Junín ,
Buenos Aires , Argentina See also ( edit ) CANAMEX Corridor Continental 1 Trans - Siberian Highway Buenos Aires References ( edit ) Jump up ^ French : Route /
Autoroute Panaméricaine / Transaméricaine , Portuguese : Rodovia / Auto - estrada Pan-americana , Spanish : Autopista / Carretera / Ruta Panamericana Jump up ^
`` A Gap in the Andes : Image of the Day '' . earthobservatory.nasa.gov. 2015 - 04 - 02 .
</item 3>

<item 4>
Pan-American Highway
However , because of the Darién Gap , it is not possible to cross between South America and Central America with conventional highway vehicles . Without an all
- terrain vehicle , it is necessary to circumnavigate this terrestrial stretch by sea . The Pan-American Highway passes through many diverse climates and
ecological types , from dense jungles , to arid deserts , some of which are passable only during the dry season , and in many regions driving is occasionally
hazardous .
</item 4>

<item 5>
Pan-American Highway
The Independent ( UK ) . Retrieved 3 August 2010 . Sources ( edit ) Plan Federal Highway System , New York Times May 15 , 1932 page XX7 Reported from the Motor
World , New York Times January 26 , 1936 page XX6 Hemisphere Road is Nearer Reality , New York Times January 7 , 1953 page 58 1997 -- 98 AAA Caribbean , Central
America and South America map Popular Mechanics , March 1943 , Longest Road In The World External links ( edit ) Media related to Pan-American Highway at
Wikimedia Commons Pan-American Highway travel guide from Wikivoyage Geographic data related to Pan-American Highway at OpenStreetMap Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Pan-American_Highway&oldid=838034809 '' Categories : Pan-American Highway Hidden categories : Articles containing
French - language text Articles containing Portuguese - language text Articles containing Spanish - language text Webarchive template wayback links CS1 Spanish
- language sources ( es ) All articles with dead external links Articles with dead external links from March 2018 Articles with permanently dead external links
Pages using div col with deprecated parameters All articles with unsourced statements Articles with
</item 5>

<item 6>
Pan-American Highway
The first conference regarding construction of the highway occurred on October 5 , 1925 . Finally , on July 29 , 1937 , Argentina , Bolivia , Chile , Colombia ,
Costa Rica , El Salvador , Guatemala , Honduras , Mexico , Nicaragua , Panama , Peru , Canada , and the United States signed the Convention on the Pan-American
Highway , whereby they agreed to speedy construction , by all adequate means . In 1950 , Mexico became the first Latin American country to complete its portion
of the highway . Countries served ( edit ) Map of the Alaska Highway portion ( in red ) of the Pan-American Highway system . The Northern Pan-American Highway
travels through 9 countries : Canada ( unofficial ) United States ( interstate system official ) Mexico Guatemala El Salvador Honduras Nicaragua Costa Rica
Panama The Southern Pan-American Highway travels through 8 countries : Suriname Guyana Venezuela Colombia Ecuador Peru Chile Argentina Important spurs also lead
into 4 countries : Bolivia Brazil Paraguay Uruguay Hemispheric overview ( edit ) The Pan-American Highway system is physically mostly complete and extends in de
facto terms from Prudhoe Bay , Alaska , in North America to the lower reaches of South America .
</item 6>

<item 7>
Pan-American Highway
Pan-American Highway - wikipedia Pan-American Highway Jump to : navigation , search `` Panamericana '' redirects here . For the film , see Panamericana ( film )
. Not to be confused with CANAMEX Corridor . The Pan-American Highway from Prudhoe Bay , U.S.A. to Quellón , Chile and Ushuaia , Argentina , with official and
unofficial routes shown in South America , and a few selected unofficial routes shown through the United States and Canada as they existed in the early 1960 's
. In 1966 the new U.S. Interstate highway system brought official status to most previously unofficial routes in the lower 48 states . The Pan-American Highway
is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) ,
called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World
Records , the Pan-American Highway is the world 's longest `` motorable road '' .
</item 7>

<item 8>
Pan-American Highway
They barely succeeded in thrashing a passage through the extreme terrain . In 1979 a team led by Mark Smith drove standard production CJ7 - model Jeeps from
South to North , traversing the Gap - with difficulty . In June 1984 , Loren and Patty Upton took 741 days slogging , winching , chopping and digging their way
through the inhospitable jungles of the Darién Gap . One proposed option to bridge the gap is a short ferry link from Colombia to a new ferry port in Panama ,
with an extension of the existing Panama highway that would complete the highway without violating these environmental concerns . Southern section ( edit ) Main
article : Pan-American Highway ( South America ) A Vía PanAm shield sign is sometimes found on routes in South American countries ( such as Chile ) associated
with the Pan-American Highway . Sculpture of a native man standing at the entrance of Fusagasugá , Colombia , over the Highway 40 . Colombia and Venezuela (
edit ) The southern part of the highway begins in northwestern Colombia , from where it follows Colombia Highway 62 to Medellín .
</item 8>

<item 9>
Pan-American Highway
Contents ( hide ) 1 Development and construction 2 Countries served 3 Hemispheric overview 4 Northern section 4.1 Alaska , Canada and the contiguous 48 states
of the United States 4.2 Mexico 4.3 Central America 5 Darién Gap 6 Southern section 6.1 Colombia and Venezuela 6.2 Ecuador , Peru , Chile , Argentina and
Paraguay 6.3 Brazil and Uruguay 6.4 Guyana , Suriname and French Guyana 7 West Indies section 8 Art and culture 9 Photo gallery 10 See also 11 References 12
Sources 13 External links Development and construction ( edit ) The concept of an overland route from one tip of the Americas to the other was originally
proposed at the First Pan-American Conference in 1889 as a railroad ; however , this proposal was never realized . The idea of building a highway emerged at the
Fifth International Conference of American States in 1923 . The first conference regarding construction of the highway occurred on October 5 , 1925 .
</item 9>

<item 10>
Pan-American Highway
In 2003 , Kevin Sanders , a long - distance rider , broke the Guinness World Record for the fastest traversal of the highway by motorcycle in 34 days . Photo
gallery ( edit ) The northern end of the Pan-American Highway at Deadhorse , Alaska Interstate 25 ( Pan-American Freeway ) approaching the Big I interchange in
Albuquerque , New Mexico , USA Pan-American Highway through San Martin , El Salvador . Another view of the Pan-American Highway in El Salvador . Pan-American
Highway in El Salvador between Lourdes and Santa Ana ; this flat 1.5 km long straight section can be used as an airstrip and it was used during El Salvador
Civil War . Highway in Guanacaste , Costa Rica ( going towards the Nicaraguan border , still many kilometres away . ) Pan-American Highway in Tres Rios , Costa
Rica , right before the toll plaza ( about 337 more km to go until the Panamanian border ) .
</item 10>
=== END ITEM id=f51977de1cedfa1d ===

=== ITEM id=297c00bd2b1b9183 ===
Gold sentences and facts:
Sentence 0: ABO and Rh blood type distribution by country ( population averages ) Country Population O+ A+ B+ AB+ O − A − B − AB − Argentina 44,270,440 45.4 % 34.26 % 8.59 % 2.64 % 8.4 % 0.44 % 0.21 % 0.06 % Armenia 2,931,568 29.0 % 46.3 % 12.0 % 5.6 % 2.0 % 3.7 % 1.0 % 0.4 % Australia 24,642,693 40.0 % 31.0 % 8.0 % 2.0 % 9.0 % 7.0 % 2.0 % 1.0 % Austria 8,592,470 30.0 % 33.0 % 12.0 % 6.0 % 7.0 % 8.0 % 3.0 % 1.0 % Bahrain 1,418,695 48.48 % 19.35 % 22.61 % 3.67 % 3.27 % 1.33 % 1.04 % 0.25 % Bangladesh 164,833,667 31.18 % 21.44 % 34.58 % 8.85 % 1.39 % 0.96 % 0.96 % 0.64 % Belgium 11,444,053 38.0 % 34.0 % 8.6 % 4.1 % 7.0 % 6.0 % 1.5 % 0.8 % Bolivia 11,053,376 51.53 % 29.45 % 10.11 % 1.15 % 4.39 % 2.73 % 0.54 % 0.1 % Bosnia and Herzegovina 3,792,730 31.0 % 36.0 % 12.0 % 6.0 % 5.0 % 7.0 % 2.0 % 1.0 % Brazil 211,248,418 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Bulgaria 7,045,097 28.0 % 37.0 % 13.0 % 7.0 % 5.0 % 7.0 % 2.0 % 1.0 % Cambodia 16,077,172 46.7 % 27.2 % 18.5 % 4.9 % 1.3 % 0.8 % 0.5 % 0.1 % Cameroon 24,515,533 42.8 % 38.8 % 12.0 % 3.3 % 1.4 % 1.2 % 0.4 % 0.1 % Canada 36,627,140 39.0 % 36.0 % 7.6 % 2.5 % 7.0 % 6.0 % 1.4 % 0.5 % Chile 18,314,060 85.5 % 8.7 % 3.35 % 1.0 % 1.2 % 0.1 % 0.05 % 0.1 % China 1,388,251,023 47.7 % 27.8 % 18.9 % 5.0 % 0.28 % 0.19 % 0.1 % 0.03 % Colombia 49,069,267 61.3 % 26.11 % 2.28 % 1.47 % 5.13 % 2.7 % 0.7 % 0.31 % Côte d'Ivoire 23,869,656 46.5 % 22.5 % 22.5 % 4.3 % 2.0 % 1.0 % 1.0 % 0.2 % Croatia 4,207,355 29.0 % 36.0 % 15.0 % 5.0 % 5.0 % 6.0 % 3.0 % 1.0 % Cuba 11,486,750 45.8 % 33.5 % 10.2 % 2.9 % 3.6 % 2.8 % 1.0 % 0.2 % Cyprus 1,189,395 35.22 % 40.35 % 11.11 % 4.72 % 3.85 % 3.48 % 0.87 % 0.40 % Czech Republic 10,555,152 27.0 % 36.0 % 15.0 % 7.0 % 5.0 % 6.0 % 3.0 % 1.0 % Denmark 5,711,902 35.0 % 37.0 % 8.0 % 4.0 % 6.0 % 7.0 % 2.0 % 1.0 % Dominican Republic 10,766,932 46.2 % 26.4 % 16.9 % 3.1 % 3.7 % 2.1 % 1.4 % 0.2 % Ecuador 16,665,756 75.0 % 14.0 % 7.1 % 0.5 % 2.38 % 0.7 % 0.3 % 0.02 % Egypt 95,220,838 52.0 % 24.0 % 12.4 % 3.8 % 5.0 % 2.0 % 0.6 % 0.2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia 104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 % 37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Greece 10,892,849 37.4 % 32.9 % 11.0 % 3.7 % 7.0 % 5.0 % 2.0 % 1.0 % Guinea 13,291,741 46.88 % 21.64 % 22.86 % 4.52 % 2.0 % 0.9 % 1.0 % 0.2 % Honduras 9,348,898 57.5 % 27.0 % 7.8 % 2.5 % 2.7 % 1.7 % 0.6 % 0.2 % Hong Kong 7,402,115 41.5 % 26.13 % 25.34 % 6.35 % 0.32 % 0.17 % 0.14 % 0.05 % Hungary 9,787,801 27.0 % 33.0 % 16.0 % 8.0 % 5.0 % 7.0 % 3.0 % 1.0 % Iceland 334,311 47.6 % 26.4 % 9.3 % 1.6 % 8.4 % 4.6 % 1.7 % 0.4 % India 1,342,561,902 27.85 % 20.8 % 38.14 % 8.93 % 1.43 % 0.57 % 1.79 % 0.49 % Indonesia 263,519,317 36.82 % 25.87 % 28.85 % 7.96 % 0.18 % 0.13 % 0.15 % 0.04 % Iran 80,982,137 33.5 % 27.0 % 22.2 % 7.0 % 4.0 % 3.0 % 2.5 % 0.8 % Iraq 38,657,787 32.1 % 25.0 % 25.6 % 7.4 % 3.6 % 2.7 % 2.7 % 0.9 % Ireland 4,749,263 47.0 % 26.0 % 9.0 % 2.0 % 8.0 % 5.0 % 2.0 % 1.0 % Israel 8,323,659 32.0 % 34.0 % 17.0 % 7.0 % 3.0 % 4.0 % 2.0 % 1.0 % Italy 59,797,969 39.0 % 36.0 % 7.5 % 2.5 % 7.0 % 6.0 % 1.5 % 0.5 % Jamaica 2,813,316 47.0 % 23.0 % 20.0 % 3.0 % 3.5 % 2.0 % 1.0 % 0.5 % Japan 126,044,340 29.9 % 39.8 % 19.9 % 9.9 % 0.15 % 0.2 % 0.1 % 0.05 % Kenya 48,470,780 45.6 % 25.2 % 21.28 % 4.2 % 1.8 % 1.0 % 0.9 % 0.02 % Latvia 1,944,122 30.6 % 31.0 % 17.0 % 6.0 % 5.4 % 6.0 % 3.0 % 1.0 % Lebanon 6,039,438 38.4 % 32.3 % 9.5 % 3.2 % 7.7 % 6.5 % 1.7 % 0.7 % Libya 6,408,990 42.64 % 28.86 % 11.19 % 4.5 % 6.26 % 4.24 % 1.64 % 0.67 % Lithuania 2,827,573 36.0 % 33.0 % 11.0 % 4.0 % 7.0 % 6.0 % 2.0 % 1.0 % Luxemburg 587,297 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Macao 606,413 41.5 % 26.1 % 25.4 % 6.3 % 0.33 % 0.17 % 0.15 % 0.05 % Macedonia 2,083,690 30.0 % 34.0 % 15.0 % 6.0 % 5.0 % 6.0 % 3.0 % 1.0 % Malaysia 31,165,480 34.32 % 30.35 % 27.37 % 7.46 % 0.17 % 0.15 % 0.14 % 0.04 % Mauritania 4,266,765 46.3 % 26.68 % 17.47 % 3.85 % 2.8 % 1.6 % 1.1 % 0.2 % Mauritius 1,267,303 38.3 % 26.0 % 25.0 % 6.7 % 1.7 % 1.0 % 1.0 % 0.3 % Mexico 130,227,836 55.79 % 29.9 % 8.0 % 1.63 % 2.7 % 1.5 % 0.4 % 0.08 % Mongolia 3,052,044 55.4 % 27.7 % 10.9 % 4.96 % 0.6 % 0.3 % 0.1 % 0.04 % Myanmar 53,507,932 35.7 % 23.8 % 32.7 % 6.95 % 0.3 % 0.2 % 0.3 % 0.05 % Nepal 29,188,100 35.2 % 28.3 % 27.1 % 8.6 % 0.3 % 0.2 % 0.2 % 0.1 % Netherlands 17,033,012 39.5 % 35.0 % 6.7 % 2.5 % 7.5 % 7.0 % 1.3 % 0.5 % New Zealand 4,604,996 38.0 % 32.0 % 9.0 % 3.0 % 9.0 % 6.0 % 2.0 % 1.0 % Nigeria 191,851,411 51.3 % 22.4 % 20.7 % 2.6 % 1.6 % 0.7 % 0.6 % 0.1 % Norway 5,330,986 33.0 % 41.5 % 6.8 % 3.4 % 6.0 % 7.5 % 1.2 % 0.6 % North Korea 25,432,033 27.15 % 31.08 % 30.15 % 11.32 % 0.08 % 0.1 % 0.1 % 0.03 % Pakistan 180,440,005 24.63 % 20.6 % 34.40 % 9.52 % 4.17 % 2.66 % 3.57 % 0.45 % Papua New Guinea 7,952,511 48.0 % 37.2 % 9.6 % 2.8 % 1.0 % 0.8 % 0.4 % 0.2 % Peru 32,167,717 70.0 % 18.4 % 7.8 % 1.6 % 1.4 % 0.5 % 0.28 % 0.02 % Philippines 103,801,747 36.9 % 28.9 % 26.9 % 6.97 % 0.1 % 0.1 % 0.1 % 0.03 % Poland 38,563,480 31.0 % 32.0 % 15.0 % 7.0 % 6.0 % 6.0 % 2.0 % 1.0 % Portugal 10,264,672 36.3 % 40.0 % 6.6 % 2.9 % 6.0 % 6.6 % 1.1 % 0.5 % Republic of Moldova 4,043,722 28.5 % 31.8 % 17.6 % 7.0 % 5.0 % 6.0 % 3.0 % 1.1 % Romania 19,237,087 28.0 % 37.0 % 14.0 % 7.0 % 5.0 % 6.0 % 2.0 % 1.0 % Russia 143,374,801 28.0 % 30.0 % 20.0 % 7.0 % 4.9 % 5.8 % 3.2 % 1.1 % Saudi Arabia 32,744,532 47.8 % 23.9 % 17.0 % 4.0 % 4.0 % 2.0 % 1.0 % 0.3 % Serbia 8,776,827 31.92 % 35.28 % 12.6 % 4.2 % 6.08 % 6.72 % 2.4 % 0.8 % Singapore 5,784,819 43.6 % 23.9 % 24.4 % 6.0 % 0.9 % 0.7 % 0.4 % 0.1 % Slovakia 5,432,265 27.2 % 35.7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0 % 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 % 7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1 % 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.8 % 4.97 % 0.2 % 0.1 % 0.2 % 0.03 % Turkey 80,420,065 29.8 % 37.8 % 14.2 % 7.2 % 3.9 % 4.7 % 1.6 % 0.8 % Uganda 43,276,492 43.7 % 39.0 % 10.7 % 3.9 % 1.3 % 1.0 % 0.3 % 0.1 % Ukraine 44,404,078 32.0 % 34.0 % 15.0 % 5.0 % 5.0 % 6.0 % 2.0 % 1.0 % United Arab Emirates 3,398,017 44.1 % 21.9 % 20.9 % 4.3 % 4.3 % 2.1 % 2.0 % 0.4 % United Kingdom 65,512,375 37.0 % 35.0 % 8.0 % 3.0 % 7.0 % 7.0 % 2.0 % 1.0 % United States 326,481,533 37.4 % 35.7 % 8.5 % 3.4 % 6.6 % 6.3 % 1.5 % 0.6 % Venezuela 32,011,082 58.5 % 28.3 % 5.3 % 1.8 % 3.5 % 1.7 % 0.7 % 0.2 % Vietnam 95,414,640 41.7 % 21.9 % 30.8 % 4.98 % 0.3 % 0.1 % 0.2 % 0.02 % Zimbabwe 16,642,880 63.3 % 18.5 % 14.6 % 0.99 % 1.7 % 0.5 % 0.4 % 0.01 % World 6,635,281,257 38.67 % 27.42 % 22.02 % 5.88 % 2.55 % 1.99 % 1.11 % 0.36 %
  Fact 0.0: The table gives ABO and Rh blood type distribution by country as population averages.
  Fact 0.1: Each row lists a country, its population, and the percentages of O+, A+, B+, AB+, O−, A−, B−, and AB− blood types.
  Fact 0.2: Argentina has a population of 44,270,440, with blood type percentages O+ 45.4, A+ 34.26, B+ 8.59, AB+ 2.64, O− 8.4, A− 0.44, B− 0.21, AB− 0.06.
  Fact 0.3: Armenia has a population of 2,931,568, with blood type percentages O+ 29.0, A+ 46.3, B+ 12.0, AB+ 5.6, O− 2.0, A− 3.7, B− 1.0, AB− 0.4.
  Fact 0.4: Australia has a population of 24,642,693, with blood type percentages O+ 40.0, A+ 31.0, B+ 8.0, AB+ 2.0, O− 9.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.5: Austria has a population of 8,592,470, with blood type percentages O+ 30.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 8.0, B− 3.0, AB− 1.0.
  Fact 0.6: Bahrain has a population of 1,418,695, with blood type percentages O+ 48.48, A+ 19.35, B+ 22.61, AB+ 3.67, O− 3.27, A− 1.33, B− 1.04, AB− 0.25.
  Fact 0.7: Bangladesh has a population of 164,833,667, with blood type percentages O+ 31.18, A+ 21.44, B+ 34.58, AB+ 8.85, O− 1.39, A− 0.96, B− 0.96, AB− 0.64.
  Fact 0.8: Belgium has a population of 11,444,053, with blood type percentages O+ 38.0, A+ 34.0, B+ 8.6, AB+ 4.1, O− 7.0, A− 6.0, B− 1.5, AB− 0.8.
  Fact 0.9: Bolivia has a population of 11,053,376, with blood type percentages O+ 51.53, A+ 29.45, B+ 10.11, AB+ 1.15, O− 4.39, A− 2.73, B− 0.54, AB− 0.1.
  Fact 0.10: Bosnia and Herzegovina has a population of 3,792,730, with blood type percentages O+ 31.0, A+ 36.0, B+ 12.0, AB+ 6.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.11: Brazil has a population of 211,248,418, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.12: Bulgaria has a population of 7,045,097, with blood type percentages O+ 28.0, A+ 37.0, B+ 13.0, AB+ 7.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.13: Cambodia has a population of 16,077,172, with blood type percentages O+ 46.7, A+ 27.2, B+ 18.5, AB+ 4.9, O− 1.3, A− 0.8, B− 0.5, AB− 0.1.
  Fact 0.14: Cameroon has a population of 24,515,533, with blood type percentages O+ 42.8, A+ 38.8, B+ 12.0, AB+ 3.3, O− 1.4, A− 1.2, B− 0.4, AB− 0.1.
  Fact 0.15: Canada has a population of 36,627,140, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.6, AB+ 2.5, O− 7.0, A− 6.0, B− 1.4, AB− 0.5.
  Fact 0.16: Chile has a population of 18,314,060, with blood type percentages O+ 85.5, A+ 8.7, B+ 3.35, AB+ 1.0, O− 1.2, A− 0.1, B− 0.05, AB− 0.1.
  Fact 0.17: China has a population of 1,388,251,023, with blood type percentages O+ 47.7, A+ 27.8, B+ 18.9, AB+ 5.0, O− 0.28, A− 0.19, B− 0.1, AB− 0.03.
  Fact 0.18: Colombia has a population of 49,069,267, with blood type percentages O+ 61.3, A+ 26.11, B+ 2.28, AB+ 1.47, O− 5.13, A− 2.7, B− 0.7, AB− 0.31.
  Fact 0.19: Côte d'Ivoire has a population of 23,869,656, with blood type percentages O+ 46.5, A+ 22.5, B+ 22.5, AB+ 4.3, O− 2.0, A− 1.0, B− 1.0, AB− 0.2.
  Fact 0.20: Croatia has a population of 4,207,355, with blood type percentages O+ 29.0, A+ 36.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.21: Cuba has a population of 11,486,750, with blood type percentages O+ 45.8, A+ 33.5, B+ 10.2, AB+ 2.9, O− 3.6, A− 2.8, B− 1.0, AB− 0.2.
  Fact 0.22: Cyprus has a population of 1,189,395, with blood type percentages O+ 35.22, A+ 40.35, B+ 11.11, AB+ 4.72, O− 3.85, A− 3.48, B− 0.87, AB− 0.40.
  Fact 0.23: Czech Republic has a population of 10,555,152, with blood type percentages O+ 27.0, A+ 36.0, B+ 15.0, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.24: Denmark has a population of 5,711,902, with blood type percentages O+ 35.0, A+ 37.0, B+ 8.0, AB+ 4.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.25: Dominican Republic has a population of 10,766,932, with blood type percentages O+ 46.2, A+ 26.4, B+ 16.9, AB+ 3.1, O− 3.7, A− 2.1, B− 1.4, AB− 0.2.
  Fact 0.26: Ecuador has a population of 16,665,756, with blood type percentages O+ 75.0, A+ 14.0, B+ 7.1, AB+ 0.5, O− 2.38, A− 0.7, B− 0.3, AB− 0.02.
  Fact 0.27: Egypt has a population of 95,220,838, with blood type percentages O+ 52.0, A+ 24.0, B+ 12.4, AB+ 3.8, O− 5.0, A− 2.0, B− 0.6, AB− 0.2.
  Fact 0.28: El Salvador has a population of 6,171,483, with blood type percentages O+ 62.0, A+ 23.0, B+ 11.0, AB+ 1.0, O− 1.0, A− 1.0, B− 0.7, AB− 0.3.
  Fact 0.29: Estonia has a population of 1,305,745, with blood type percentages O+ 29.5, A+ 30.8, B+ 20.7, AB+ 6.3, O− 4.3, A− 4.5, B− 3.0, AB− 0.9.
  Fact 0.30: Ethiopia has a population of 104,352,727, with blood type percentages O+ 39.0, A+ 28.0, B+ 21.0, AB+ 5.0, O− 3.0, A− 2.0, B− 1.0, AB− 1.0.
  Fact 0.31: Finland has a population of 5,541,328, with blood type percentages O+ 27.0, A+ 38.0, B+ 15.0, AB+ 7.0, O− 4.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.32: France has a population of 64,939,560, with blood type percentages O+ 36.0, A+ 37.0, B+ 9.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.33: Germany has a population of 80,635,980, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.34: Greece has a population of 10,892,849, with blood type percentages O+ 37.4, A+ 32.9, B+ 11.0, AB+ 3.7, O− 7.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.35: Guinea has a population of 13,291,741, with blood type percentages O+ 46.88, A+ 21.64, B+ 22.86, AB+ 4.52, O− 2.0, A− 0.9, B− 1.0, AB− 0.2.
  Fact 0.36: Honduras has a population of 9,348,898, with blood type percentages O+ 57.5, A+ 27.0, B+ 7.8, AB+ 2.5, O− 2.7, A− 1.7, B− 0.6, AB− 0.2.
  Fact 0.37: Hong Kong has a population of 7,402,115, with blood type percentages O+ 41.5, A+ 26.13, B+ 25.34, AB+ 6.35, O− 0.32, A− 0.17, B− 0.14, AB− 0.05.
  Fact 0.38: Hungary has a population of 9,787,801, with blood type percentages O+ 27.0, A+ 33.0, B+ 16.0, AB+ 8.0, O− 5.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.39: Iceland has a population of 334,311, with blood type percentages O+ 47.6, A+ 26.4, B+ 9.3, AB+ 1.6, O− 8.4, A− 4.6, B− 1.7, AB− 0.4.
  Fact 0.40: India has a population of 1,342,561,902, with blood type percentages O+ 27.85, A+ 20.8, B+ 38.14, AB+ 8.93, O− 1.43, A− 0.57, B− 1.79, AB− 0.49.
  Fact 0.41: Indonesia has a population of 263,519,317, with blood type percentages O+ 36.82, A+ 25.87, B+ 28.85, AB+ 7.96, O− 0.18, A− 0.13, B− 0.15, AB− 0.04.
  Fact 0.42: Iran has a population of 80,982,137, with blood type percentages O+ 33.5, A+ 27.0, B+ 22.2, AB+ 7.0, O− 4.0, A− 3.0, B− 2.5, AB− 0.8.
  Fact 0.43: Iraq has a population of 38,657,787, with blood type percentages O+ 32.1, A+ 25.0, B+ 25.6, AB+ 7.4, O− 3.6, A− 2.7, B− 2.7, AB− 0.9.
  Fact 0.44: Ireland has a population of 4,749,263, with blood type percentages O+ 47.0, A+ 26.0, B+ 9.0, AB+ 2.0, O− 8.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.45: Israel has a population of 8,323,659, with blood type percentages O+ 32.0, A+ 34.0, B+ 17.0, AB+ 7.0, O− 3.0, A− 4.0, B− 2.0, AB− 1.0.
  Fact 0.46: Italy has a population of 59,797,969, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.5, AB+ 2.5, O− 7.0, A− 6.0, B− 1.5, AB− 0.5.
  Fact 0.47: Jamaica has a population of 2,813,316, with blood type percentages O+ 47.0, A+ 23.0, B+ 20.0, AB+ 3.0, O− 3.5, A− 2.0, B− 1.0, AB− 0.5.
  Fact 0.48: Japan has a population of 126,044,340, with blood type percentages O+ 29.9, A+ 39.8, B+ 19.9, AB+ 9.9, O− 0.15, A− 0.2, B− 0.1, AB− 0.05.
  Fact 0.49: Kenya has a population of 48,470,780, with blood type percentages O+ 45.6, A+ 25.2, B+ 21.28, AB+ 4.2, O− 1.8, A− 1.0, B− 0.9, AB− 0.02.
  Fact 0.50: Latvia has a population of 1,944,122, with blood type percentages O+ 30.6, A+ 31.0, B+ 17.0, AB+ 6.0, O− 5.4, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.51: Lebanon has a population of 6,039,438, with blood type percentages O+ 38.4, A+ 32.3, B+ 9.5, AB+ 3.2, O− 7.7, A− 6.5, B− 1.7, AB− 0.7.
  Fact 0.52: Libya has a population of 6,408,990, with blood type percentages O+ 42.64, A+ 28.86, B+ 11.19, AB+ 4.5, O− 6.26, A− 4.24, B− 1.64, AB− 0.67.
  Fact 0.53: Lithuania has a population of 2,827,573, with blood type percentages O+ 36.0, A+ 33.0, B+ 11.0, AB+ 4.0, O− 7.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.54: Luxemburg has a population of 587,297, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.55: Macao has a population of 606,413, with blood type percentages O+ 41.5, A+ 26.1, B+ 25.4, AB+ 6.3, O− 0.33, A− 0.17, B− 0.15, AB− 0.05.
  Fact 0.56: Macedonia has a population of 2,083,690, with blood type percentages O+ 30.0, A+ 34.0, B+ 15.0, AB+ 6.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.57: Malaysia has a population of 31,165,480, with blood type percentages O+ 34.32, A+ 30.35, B+ 27.37, AB+ 7.46, O− 0.17, A− 0.15, B− 0.14, AB− 0.04.
  Fact 0.58: Mauritania has a population of 4,266,765, with blood type percentages O+ 46.3, A+ 26.68, B+ 17.47, AB+ 3.85, O− 2.8, A− 1.6, B− 1.1, AB− 0.2.
  Fact 0.59: Mauritius has a population of 1,267,303, with blood type percentages O+ 38.3, A+ 26.0, B+ 25.0, AB+ 6.7, O− 1.7, A− 1.0, B− 1.0, AB− 0.3.
  Fact 0.60: Mexico has a population of 130,227,836, with blood type percentages O+ 55.79, A+ 29.9, B+ 8.0, AB+ 1.63, O− 2.7, A− 1.5, B− 0.4, AB− 0.08.
  Fact 0.61: Mongolia has a population of 3,052,044, with blood type percentages O+ 55.4, A+ 27.7, B+ 10.9, AB+ 4.96, O− 0.6, A− 0.3, B− 0.1, AB− 0.04.
  Fact 0.62: Myanmar has a population of 53,507,932, with blood type percentages O+ 35.7, A+ 23.8, B+ 32.7, AB+ 6.95, O− 0.3, A− 0.2, B− 0.3, AB− 0.05.
  Fact 0.63: Nepal has a population of 29,188,100, with blood type percentages O+ 35.2, A+ 28.3, B+ 27.1, AB+ 8.6, O− 0.3, A− 0.2, B− 0.2, AB− 0.1.
  Fact 0.64: Netherlands has a population of 17,033,012, with blood type percentages O+ 39.5, A+ 35.0, B+ 6.7, AB+ 2.5, O− 7.5, A− 7.0, B− 1.3, AB− 0.5.
  Fact 0.65: New Zealand has a population of 4,604,996, with blood type percentages O+ 38.0, A+ 32.0, B+ 9.0, AB+ 3.0, O− 9.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.66: Nigeria has a population of 191,851,411, with blood type percentages O+ 51.3, A+ 22.4, B+ 20.7, AB+ 2.6, O− 1.6, A− 0.7, B− 0.6, AB− 0.1.
  Fact 0.67: Norway has a population of 5,330,986, with blood type percentages O+ 33.0, A+ 41.5, B+ 6.8, AB+ 3.4, O− 6.0, A− 7.5, B− 1.2, AB− 0.6.
  Fact 0.68: North Korea has a population of 25,432,033, with blood type percentages O+ 27.15, A+ 31.08, B+ 30.15, AB+ 11.32, O− 0.08, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.69: Pakistan has a population of 180,440,005, with blood type percentages O+ 24.63, A+ 20.6, B+ 34.40, AB+ 9.52, O− 4.17, A− 2.66, B− 3.57, AB− 0.45.
  Fact 0.70: Papua New Guinea has a population of 7,952,511, with blood type percentages O+ 48.0, A+ 37.2, B+ 9.6, AB+ 2.8, O− 1.0, A− 0.8, B− 0.4, AB− 0.2.
  Fact 0.71: Peru has a population of 32,167,717, with blood type percentages O+ 70.0, A+ 18.4, B+ 7.8, AB+ 1.6, O− 1.4, A− 0.5, B− 0.28, AB− 0.02.
  Fact 0.72: Philippines has a population of 103,801,747, with blood type percentages O+ 36.9, A+ 28.9, B+ 26.9, AB+ 6.97, O− 0.1, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.73: Poland has a population of 38,563,480, with blood type percentages O+ 31.0, A+ 32.0, B+ 15.0, AB+ 7.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.74: Portugal has a population of 10,264,672, with blood type percentages O+ 36.3, A+ 40.0, B+ 6.6, AB+ 2.9, O− 6.0, A− 6.6, B− 1.1, AB− 0.5.
  Fact 0.75: Republic of Moldova has a population of 4,043,722, with blood type percentages O+ 28.5, A+ 31.8, B+ 17.6, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.1.
  Fact 0.76: Romania has a population of 19,237,087, with blood type percentages O+ 28.0, A+ 37.0, B+ 14.0, AB+ 7.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.77: Russia has a population of 143,374,801, with blood type percentages O+ 28.0, A+ 30.0, B+ 20.0, AB+ 7.0, O− 4.9, A− 5.8, B− 3.2, AB− 1.1.
  Fact 0.78: Saudi Arabia has a population of 32,744,532, with blood type percentages O+ 47.8, A+ 23.9, B+ 17.0, AB+ 4.0, O− 4.0, A− 2.0, B− 1.0, AB− 0.3.
  Fact 0.79: Serbia has a population of 8,776,827, with blood type percentages O+ 31.92, A+ 35.28, B+ 12.6, AB+ 4.2, O− 6.08, A− 6.72, B− 2.4, AB− 0.8.
  Fact 0.80: Singapore has a population of 5,784,819, with blood type percentages O+ 43.6, A+ 23.9, B+ 24.4, AB+ 6.0, O− 0.9, A− 0.7, B− 0.4, AB− 0.1.
  Fact 0.81: Slovakia has a population of 5,432,265, with blood type percentages O+ 27.2, A+ 35.7, B+ 15.3, AB+ 6.8, O− 4.8, A− 6.3, B− 2.7, AB− 1.2.
  Fact 0.82: Slovenia has a population of 2,071,258, with blood type percentages O+ 31.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.83: South Africa has a population of 55,437,815, with blood type percentages O+ 39.0, A+ 32.0, B+ 12.0, AB+ 3.0, O− 6.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.84: South Korea has a population of 50,748,307, with blood type percentages O+ 27.9, A+ 33.87, B+ 26.92, AB+ 10.98, O− 0.1, A− 0.13, B− 0.08, AB− 0.02.
  Fact 0.85: Spain has a population of 46,070,163, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.86: Sudan has a population of 40,827,975, with blood type percentages O+ 48.0, A+ 27.7, B+ 15.2, AB+ 2.8, O− 3.5, A− 1.8, B− 0.8, AB− 0.2.
  Fact 0.87: Sweden has a population of 9,920,843, with blood type percentages O+ 32.0, A+ 37.0, B+ 10.0, AB+ 5.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.88: Switzerland has a population of 8,454,321, with blood type percentages O+ 35.0, A+ 40.0, B+ 7.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.89: Syria has a population of 18,920,698, with blood type percentages O+ 43.0, A+ 30.0, B+ 14.0, AB+ 3.7, O− 5.0, A− 3.0, B− 1.0, AB− 0.3.
  Fact 0.90: Taiwan has a population of 23,234,936, with blood type percentages O+ 43.9, A+ 25.9, B+ 23.9, AB+ 6.0, O− 0.17, A− 0.1, B− 0.01, AB− 0.02.
  Fact 0.91: Thailand has a population of 68,298,027, with blood type percentages O+ 40.8, A+ 16.9, B+ 36.8, AB+ 4.97, O− 0.2, A− 0.1, B− 0.2, AB− 0.03.
  Fact 0.92: Turkey has a population of 80,420,065, with blood type percentages O+ 29.8, A+ 37.8, B+ 14.2, AB+ 7.2, O− 3.9, A− 4.7, B− 1.6, AB− 0.8.
  Fact 0.93: Uganda has a population of 43,276,492, with blood type percentages O+ 43.7, A+ 39.0, B+ 10.7, AB+ 3.9, O− 1.3, A− 1.0, B− 0.3, AB− 0.1.
  Fact 0.94: Ukraine has a population of 44,404,078, with blood type percentages O+ 32.0, A+ 34.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.95: United Arab Emirates has a population of 3,398,017, with blood type percentages O+ 44.1, A+ 21.9, B+ 20.9, AB+ 4.3, O− 4.3, A− 2.1, B− 2.0, AB− 0.4.
  Fact 0.96: United Kingdom has a population of 65,512,375, with blood type percentages O+ 37.0, A+ 35.0, B+ 8.0, AB+ 3.0, O− 7.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.97: United States has a population of 326,481,533, with blood type percentages O+ 37.4, A+ 35.7, B+ 8.5, AB+ 3.4, O− 6.6, A− 6.3, B− 1.5, AB− 0.6.
  Fact 0.98: Venezuela has a population of 32,011,082, with blood type percentages O+ 58.5, A+ 28.3, B+ 5.3, AB+ 1.8, O− 3.5, A− 1.7, B− 0.7, AB− 0.2.
  Fact 0.99: Vietnam has a population of 95,414,640, with blood type percentages O+ 41.7, A+ 21.9, B+ 30.8, AB+ 4.98, O− 0.3, A− 0.1, B− 0.2, AB− 0.02.
  Fact 0.100: Zimbabwe has a population of 16,642,880, with blood type percentages O+ 63.3, A+ 18.5, B+ 14.6, AB+ 0.99, O− 1.7, A− 0.5, B− 0.4, AB− 0.01.
  Fact 0.101: The World row has a population of 6,635,281,257, with blood type percentages O+ 38.67, A+ 27.42, B+ 22.02, AB+ 5.88, O− 2.55, A− 1.99, B− 1.11, AB− 0.36.

Retrieved items:
<item 1>
Blood type distribution by country
7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0
% 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0
% 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.
</item 1>

<item 2>
Blood type distribution by country
5 % 0.28 % 0.02 % Philippines 103,801,747 36.9 % 28.9 % 26.9 % 6.97 % 0.1 % 0.1 % 0.1 % 0.03 % Poland 38,563,480 31.0 % 32.0 % 15.0 % 7.0 % 6.0 % 6.0 % 2.0 %
1.0 % Portugal 10,264,672 36.3 % 40.0 % 6.6 % 2.9 % 6.0 % 6.6 % 1.1 % 0.5 % Republic of Moldova 4,043,722 28.5 % 31.8 % 17.6 % 7.0 % 5.0 % 6.0 % 3.0 % 1.1 %
Romania 19,237,087 28.0 % 37.0 % 14.0 % 7.0 % 5.0 % 6.0 % 2.0 % 1.
</item 2>

<item 3>
Blood type distribution by country
975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 %
7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1
% 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.
</item 3>

<item 4>
Blood type distribution by country
Retrieved 2017 - 11 - 23 . Jump up ^ `` Veregruppidest '' ( Blood groups ) . verekeskus.ee ( in Estonian ) . Retrieved 2017 - 11 - 23 . Jump up ^ ``
Suomalaisten veriryhmäjakauma '' ( in Finnish ) . 2009 - 08 - 21 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Les groupes sanguins ( système ABO ) '' . Centre
Hospitalier Princesse GRACE - Monaco ( in French ) . C.H.P.G. MONACO. 2005 . Retrieved 2008 - 07 - 15 . Jump up ^ `` Blóðflokkar '' ( in Icelandic ) ...
landspitali.is . Archived from the original on 2011 - 07 - 19 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Blood Type Frequencies by Country including the Rh
Factor '' .
</item 4>

<item 5>
Blood type distribution by country
04 % Myanmar 53,507,932 35.7 % 23.8 % 32.7 % 6.95 % 0.3 % 0.2 % 0.3 % 0.05 % Nepal 29,188,100 35.2 % 28.3 % 27.1 % 8.6 % 0.3 % 0.2 % 0.2 % 0.1 % Netherlands
17,033,012 39.5 % 35.0 % 6.7 % 2.5 % 7.5 % 7.0 % 1.3 % 0.5 % New Zealand 4,604,996 38.0 % 32.0 % 9.0 % 3.0 % 9.0 % 6.0 % 2.0 % 1.0 % Nigeria 191,851,411 51.3 %
22.4 % 20.7 % 2.6 % 1.6 % 0.7 % 0.6 % 0.1 % Norway 5,330,986 33.
</item 5>

<item 6>
Blood type distribution by country
0 % 1.0 % Germany 80,635,980 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Greece 10,892,849 37.4 % 32.9 % 11.0 % 3.7 % 7.0 % 5.0 % 2.0 % 1.0 % Guinea
13,291,741 46.88 % 21.64 % 22.86 % 4.52 % 2.0 % 0.9 % 1.0 % 0.2 % Honduras 9,348,898 57.5 % 27.0 % 7.8 % 2.5 % 2.7 % 1.7 % 0.6 % 0.2 % Hong Kong 7,402,115 41.5
% 26.13 % 25.34 % 6.35 % 0.32 % 0.17 % 0.14 % 0.05 % Hungary 9,787,
</item 6>

<item 7>
Blood type distribution by country
0 % 1.5 % 0.8 % Bolivia 11,053,376 51.53 % 29.45 % 10.11 % 1.15 % 4.39 % 2.73 % 0.54 % 0.1 % Bosnia and Herzegovina 3,792,730 31.0 % 36.0 % 12.0 % 6.0 % 5.0 %
7.0 % 2.0 % 1.0 % Brazil 211,248,418 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Bulgaria 7,045,097 28.0 % 37.0 % 13.0 % 7.0 % 5.0 % 7.0 % 2.0 % 1.0 %
Cambodia 16,077,172 46.7 % 27.2 % 18.5 % 4.9 % 1.3 % 0.8 % 0.5 % 0.
</item 7>

<item 8>
Blood type distribution by country
In countries such as Austria , Denmark , Norway , and Switzerland , about 45 -- 50 % of the population have this blood type , whereas about 40 % of Poles and
Ukrainians do so . The highest frequencies are found in small , unrelated populations . For example , about 80 % of the Blackfoot Indians of Montana have blood
type A ... Jump up ^ Technical Monograph No. 2 : The ABO Blood Group System and ABO Subgroups ( PDF ) . Biotec . March 2005 . Archived from the original ( PDF )
on 6 February 2007 ... The frequency of blood group A is quite high ( 25 -- 55 % ) in Europe , especially in Scandinavia and parts of central Europe . High
group A frequency is also found in the Aboriginal people of South Australia ( up to 45 % ) and in certain American Indian tribes where the frequency reaches 35
% .
</item 8>

<item 9>
Blood type distribution by country
40 11 Spanish 38 47 10 5 Sudanese 62 16 21 0 Swedish 38 47 10 5 Swiss 40 50 7 Tatars 28 30 29 13 Thais 37 22 33 8 Turks 43 34 18 6 Ukrainians 37 40 18 6 Black
Americans 49 27 20 White Americans 45 40 11 Vietnamese 42 22 30 5 Mean 43.91 34.80 16.55 5.14 Standard deviation 16.87 13.80 9.97 3.41 Blood group B has its
highest frequency in South Asia where it ranks first as the largest share of the earth 's population . In Southeast Asia the share of the population is high ,
especially in Thailand and Indonesia , then in East Asia , Northern Asia and neighboring Central Asia , and its incidence diminishes both towards the east and
the west , falling to single digit percentages in Netherlands , Norway , Portugal and Switzerland . It is believed to have been entirely absent from Native
American and Australian Aboriginal populations prior to the arrival of Europeans in those areas .
</item 9>

<item 10>
Blood type distribution by country
Retrieved 2010 - 11 - 19 . Jump up ^ `` Federación Nacional de Donantes de Sangre / La sangre / Grupos '' . Donantesdesangre.net . Retrieved 2010 - 11 - 19 .
Jump up ^ `` Frequency of major blood groups in the Swedish population '' . Geblod.nu. 2007 - 10 - 02 . Archived from the original on 24 November 2010 .
Retrieved 2010 - 11 - 19 . Jump up ^ - BLUTSPENDE https://www.blutspende.ch/fr/don_de_sang/sanguin_groupe/groupe_sanguin - BLUTSPENDE Check url = value ( help )
. Missing or empty title = ( help ) Jump up ^ `` Turkey Blood Group Site '' . Kangrubu.com . Archived from the original on 29 May 2010 . Retrieved 2010 - 11 -
19 . Jump up ^ `` Blood Group Basics '' . Blood.co.uk . Retrieved 2017 - 07 - 30 .
</item 10>
=== END ITEM id=297c00bd2b1b9183 ===

=== ITEM id=d86192cf676b95a5 ===
Gold sentences and facts:
Sentence 0: With the Intertropical Convergence Zone ( ITCZ ) swinging northward over West Africa from the Southern Hemisphere in April , heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical ( south westerlies ) and the Continental tropical ( north easterlies ) , begins in central Nigeria while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity , heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in central Nigeria varies from 1,100 mm ( 43.3 in ) in the lowlands of the river Niger Benue trough to over 2,000 mm ( 78.7 in ) along the south western escarpment of the Jos Plateau .
  Fact 0.0: The Intertropical Convergence Zone (ITCZ) swings northward over West Africa from the Southern Hemisphere in April.
  Fact 0.1: In April, heavy showers come from pre-monsoonal convective clouds, mainly in the form of squall lines.
  Fact 0.2: The squall lines are also known as the north easterlies.
  Fact 0.3: The north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria.
  Fact 0.4: The two dominant airmasses in Nigeria are the Maritime tropical (south westerlies) and the Continental tropical (north easterlies).
  Fact 0.5: The pre-monsoonal showers begin in central Nigeria.
  Fact 0.6: The monsoons from the South Atlantic Ocean arrive in central Nigeria in July.
  Fact 0.7: The monsoons bring high humidity, heavy cloud cover and heavy rainfall to central Nigeria in July.
  Fact 0.8: The rainfall from the monsoons can be a daily occurrence in central Nigeria.
  Fact 0.9: The monsoon rainfall in central Nigeria lasts until September.
  Fact 0.10: The monsoons gradually begin retreating southward to the southern part of Nigeria when September arrives.
  Fact 0.11: Rainfall totals in central Nigeria vary from 1,100 mm (43.3 in) in the lowlands of the river Niger Benue trough.
  Fact 0.12: Rainfall totals in central Nigeria reach over 2,000 mm (78.7 in) along the south western escarpment of the Jos Plateau.

Retrieved items:
<item 1>
Geography of Nigeria
In the rainy season it is damp , and the rainfalls are usually abundant . The dry season of Nigeria is a period of little cloud cover in the southern part of
Nigeria to virtually no cloud cover in the northern part of Nigeria . The sun shines through the atmosphere with little obstructions from the clear skies making
the dry season in Nigeria a period of warm weather conditions.In the middle of the dry season around December , a dusty wind from the Sahara Desert called the
harmattan enters Nigeria from the northeastern part of the country blocking sun rays partially from shining and also creating haze in the atmosphere , this
activities of the wind lowers temperatures considerably saving inhabitants for sometime , from the scorching heat that would have occurred as a result of
clearer skies during the dry season . But with the withdrawal of this wind around March to April following the onset of the rainy season , temperatures can go
as high as 44 ° C ( 111.2 ° F ) in some parts of Nigeria . Semi temperate weather conditions prevail on the highlands in central Nigeria above 1,200 metres (
3,937 ft ) above sea level , namely the Jos Plateau .
</item 1>

<item 2>
Geography of Nigeria
Temperatures in the northern part of Nigeria can go as high as 48 ° C ( 118.4 ° F ) in cities like Maiduguri . The high temperatures coupled with an increase in
insolation causes a region of low pressure to develop over west Africa and Nigeria ( between March to May ) . The Tropical continental airmass ( CT ) from the
Sahara Desert in the northern part of West Africa , is weakened due to the overheating of the land surface in west Africa and Nigeria at this time . The
Tropical continental airmass ( CT ) begins to retreat northwards to the Sahara Desert due to massive heating of the land which transfers heat in the form of
convection into the Tropical continental airmass ( CT ) which constitutes the main layer of air above the land . This transfer of heat in the Tropical
continental airmass ( CT ) in turn , causes the wind to expand and become lighter as this is the normal behaviour for winds moving above intensely heated
grounds .
</item 2>

<item 3>
Geography of Nigeria
The Tropical continental airmass ( CT ) loses its strength as a major airmass in the region of west Africa and over Nigeria at this time ( around February in
the southern part of Nigeria to June in northern Nigeria ) and begins to retreat coupled with the rising of air in form of convection within this airmass (
Tropical continental airmass ( CT ) ) , further weakening the dominance of the wind over west Africa and Nigeria . The Tropical continental airmass ( CT )
finally retreats from most part of Nigeria , and the West African atmosphere around April to May , leaving an empty atmosphere over Nigeria . The sun 's rays
enters into the atmosphere of Nigeria more intense than it does during the presence of the Tropical continental airmass , which contained dust ( in form of haze
) that reduced the intensity of the sun . The overheating of the west Africa land mass and Nigeria in particular creates a low pressure region over west Africa
and Nigeria . This low pressure zone attracts the Tropical Maritime Airmass ( MT ) from the south Atlantic Ocean since areas of low pressures experiences inward
blowing winds because winds are moving air blowing outwards from regions of high pressure to regions of low pressure .
</item 3>

<item 4>
Geography of Nigeria
The northward retreat of the tropical continental airmass ( CT ) , is caused by the sun 's northward shift from the tropic of capricorn in the southern
hemisphere to the tropic of cancer in the northern hemisphere . This shift begins from February and ends in June , when the sun is fully overhead , at the
tropic of cancer in the northern hemisphere . During this northward migration of the sun as a result of the earth tilting along its axis , the sun crosses the
equator ( around March ) , moving over west Africa at this time on its journey to the northern hemisphere . West Africa comes directly under the sun at this
time . The sun is overhead throughout west Africa and over Nigeria during this period of the sun 's northward migration to the tropic of cancer in the northern
hemisphere . The whole of West Africa is heated intensely as result of the increased insolation received from the sun being overhead over west Africa .
Temperatures can climb as high as 35 ° C ( 95.0 ° F ) over west Africa during this time .
</item 4>

<item 5>
Geography of Nigeria
These are the dry season and the rainy season . The dry season is accompanied by a dust laden airmass from the Sahara Desert , locally known as Harmattan , or
by its main name , The Tropical Continental ( CT ) airmass , while the rainy season is heavily influenced by an airmass originating from the south atlantic
ocean , locally known as the south west wind , or by its main name , The Tropical Maritime ( MT ) airmass . These two major wind systems in Nigeria are known as
the trade winds . Trade winds ( edit ) Tropical Maritime airmass ( edit ) The tropical maritime airmass ( MT ) is responsible for Nigeria 's rainy season . This
wind ( the tropical maritime airmass ) invades the country from February in the southern part of Nigeria while it takes longer for the wind to fully cover the
whole of the country , reaching the northern part of Nigeria in June . Its invasion is as a result of the northward retreat , of the tropical continental
airmass ( CT ) known as the harmattan .
</item 5>

<item 6>
Nigeria, in tropical West Africa, borders Benin, Niger, Chad and Cameroon, with a southern coast on the Gulf of Guinea and Lake Chad to the northeast. Major
features include the Niger and Benue valleys, highlands and plateaus, and the vast Niger Delta. Climate ranges from humid, exceptionally rainy southern monsoon
to central savanna and dry northern Sahel; highlands are cooler. The country has rainy and dry seasons, with Harmattan dust winds in the dry season.
</item 6>

<item 7>
Geography of Nigeria
Nigeria 's northern end is south of the 15 degrees line at about 14 degrees . Nigeria 's location in the wetter part of the easterly waves south of the 15
degree line creates wetter climatic conditions for Nigeria especially during the monsoons . Tropical continental airmass ( edit ) The Tropical Continental
Airmass ( CT ) locally known as the harmattan , is a wind originating from North Africa which crosses the Sahara Desert into west Africa to Nigeria . This
airmass dominates Nigeria 's climate during the dry season from December to March . The Tropical continental airmass is dusty and creates a haze within the
atmosphere of west Africa and Nigeria when it predominates . The haze is as a result of the dust within the airmass limiting visibility and blocking much of the
sun 's rays from reaching the earth . Effects of the tropical continental airmass ( edit ) The mass has no ability of forming rain due to low humidity within
the airmass , since it crosses the Sahara Desert , it picks up dust instead of water thereby creating little chances for rainfall .
</item 7>

<item 8>
Geography of Nigeria
With the Intertropical Convergence Zone ( ITCZ ) swinging northward over West Africa from the Southern Hemisphere in April , heavy showers coming from
pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two
dominant airmasses in Nigeria known as the Maritime tropical ( south westerlies ) and the Continental tropical ( north easterlies ) , begins in central Nigeria
while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity , heavy cloud cover and heavy rainfall which
can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in
central Nigeria varies from 1,100 mm ( 43.3 in ) in the lowlands of the river Niger Benue trough to over 2,000 mm ( 78.7 in ) along the south western escarpment
of the Jos Plateau . The Sahel climate or tropical dry climate , is the predominant climate type in the northern part of Nigeria . Annual rainfall totals are
lower compared to the southern and central part of Nigeria .
</item 8>

<item 9>
Geography of Nigeria
The Tropical monsoon climate has a very small temperature range . Then temperature ranges are almost constant throughout the year , for example , Warri town in
the southern part of Nigeria , records a maximum of 28 ° C ( 82.4 ° F ) for its hottest month while its lowest temperature is 26 ° C ( 78.8 ° F ) in its coldest
month . The temperature difference of Warri town is not more than 2 ° C ( 5 ° F ) . The southern part of Nigeria experiences heavy and abundant rainfall . These
storms are usually convectional in nature due to the regions proximity , to the equatorial belt . The annual rainfall received in this region is very high ,
usually above the 2,000 mm ( 78.7 in ) rainfall totals giving for tropical rainforest climates worldwide . Over 4,000 mm ( 157.5 in ) of rainfall is received in
the coastal region of Nigeria around the Niger delta area . Bonny town found in the coastal region of the Niger delta area in southern Nigeria receives well
over 4,000 mm ( 157.5 in ) of rainfall annually .
</item 9>

<item 10>
Geography of Nigeria
The rest of the southeast receives between 2,000 and 3,000 mm ( 118.1 in ) of rain per year . The southern region of Nigeria experiences a double rainfall
maxima characterised by two high rainfall peaks , with a short dry season and a longer dry season falling between and after each peaks . The first rainy season
begins around March and last to the end of July with a peak in June , this rainy season is followed by a short dry break in August known as the August break
which is a short dry season lasting for two to three weeks in August . This break is broken by the short rainy season starting around early September and
lasting to Mid October with a peak period at the end of September . The ending of the short rainy season in October is followed by long dry season . This period
starts from late October and lasts till early March with peak dry conditions between early December and late February . The tropical savanna climate or tropical
wet and dry climate , is extensive in area and covers most of Western Nigeria to central Nigeria beginning from the Tropical rainforest climate boundary in
southern Nigeria to the central part of Nigeria , where it exerts enormous influence on the region .
</item 10>
=== END ITEM id=d86192cf676b95a5 ===

=== ITEM id=dd6e065e12d75100 ===
Gold sentences and facts:
Sentence 0: Celebrity Notability ( known for ) Professional partner Status Trista Sutter Reality television star Louis Van Amstel Eliminated 1st on June 8 , 2005 Evander Holyfield Professional boxer Edyta Sliwinska Eliminated 2nd on June 15 , 2005 Rachel Hunter Supermodel & actress Jonathan Roberts Eliminated 3rd on June 22 , 2005 Joey McIntyre New Kids on the Block singer Ashly DelGrosso Third place on June 29 , 2005 John O'Hurley Actor & game show host Charlotte Jorgensen Runner - up on July 6 , 2005 Kelly Monaco Actress Alec Mazo Winner on July 6 , 2005
  Fact 0.0: Trista Sutter is known for being a reality television star.
  Fact 0.1: Trista Sutter's professional partner is Louis Van Amstel.
  Fact 0.2: Trista Sutter was eliminated 1st on June 8, 2005.
  Fact 0.3: Evander Holyfield is a professional boxer.
  Fact 0.4: Evander Holyfield's professional partner is Edyta Sliwinska.
  Fact 0.5: Evander Holyfield was eliminated 2nd on June 15, 2005.
  Fact 0.6: Rachel Hunter is a supermodel and actress.
  Fact 0.7: Rachel Hunter's professional partner is Jonathan Roberts.
  Fact 0.8: Rachel Hunter was eliminated 3rd on June 22, 2005.
  Fact 0.9: Joey McIntyre is a New Kids on the Block singer.
  Fact 0.10: Joey McIntyre's professional partner is Ashly DelGrosso.
  Fact 0.11: Joey McIntyre finished in third place on June 29, 2005.
  Fact 0.12: John O'Hurley is an actor and game show host.
  Fact 0.13: John O'Hurley's professional partner is Charlotte Jorgensen.
  Fact 0.14: John O'Hurley finished as runner-up on July 6, 2005.
  Fact 0.15: Kelly Monaco is an actress.
  Fact 0.16: Kelly Monaco's professional partner is Alec Mazo.
  Fact 0.17: Kelly Monaco was the winner on July 6, 2005.

Retrieved items:
<item 1>
Dancing with the Stars (U.S. season 1)
Jump up ^ `` Dancing with the Stars : Episode 105 '' . TV.com . Retrieved 2009 - 05 - 06 . Jump up ^ `` Dancing with the Stars : Episode 106 '' . TV.com .
Retrieved 2009 - 05 - 06 . External links ( edit ) Official website ( hide ) Dancing with the Stars ( U.S. TV series ) Seasons 5 6 7 8 9 10 11 12 13 14 15 16 17
18 19 20 21 22 23 24 25 Participants List of competitors Winners Celebrity Kelly Monaco Drew Lachey Emmitt Smith Apolo Anton Ohno Hélio Castroneves Kristi
Yamaguchi Brooke Burke Shawn Johnson Donny Osmond Nicole Scherzinger Jennifer Grey Hines Ward J.R.
</item 1>

<item 2>
The first U.S. season of *Dancing with the Stars* aired on ABC June 1–July 6, 2005, with six celebrity–professional pairs across eight episodes. Hosted by Tom
Bergeron and Lisa Canning, it featured judges Carrie Ann Inaba, Len Goodman, and Bruno Tonioli. Kelly Monaco and Alec Mazo won; John O’Hurley and Charlotte
Jorgensen were runners-up, and Joey McIntyre and Ashly DelGrosso placed third
</item 2>

<item 3>
Dancing with the Stars (U.S. season 1)
8 , 9 ) Freestyle `` I Am What I Am '' -- Gloria Gaynor Dance chart ( edit ) Highest scoring dance Lowest scoring dance Couple 5 6 Kelly and Alec Waltz Rumba
Jive Samba Viennese Waltz Foxtrot Paso doble Samba Freestyle John and Charlotte Cha - cha - cha Quickstep Tango Samba Viennese Waltz Foxtrot Paso doble
Quickstep Freestyle Joey and Ashly Cha - cha - cha Quickstep Jive Samba Viennese Waltz Foxtrot Paso doble Rachel and Jonathan Waltz Rumba Tango Samba Viennese
Waltz Evander and Edyta Cha - cha - cha Quickstep Jive Trista and Louis Waltz Rumba References ( edit ) Jump up ^ Associated Press ( 2005 - 06 - 08 ) . `` '
Dancing With the Stars ' Debuts in 2nd Place '' . FoxNews . Retrieved 2009 - 05 - 06 . Jump up ^ Maynard , John ( 2005 - 07 - 13 ) .
</item 3>

<item 4>
Dancing with the Stars (U.S. season 1)
The season finale aired July 6 , 2005 . Contents ( hide ) 1 Couples 2 Scoring chart 2.1 Average chart 2.2 Pro 's average chart 3 Highest and lowest scoring
performances 4 Couples ' highest and lowest scoring performances 5 Weekly scores and songs 5.1 Week 1 5.2 Week 2 5.3 Week 3 5.4 Week 4 5.5 Week 5 5.6 Week 6 :
Finale 5.7 Dance off 6 Dance chart 7 References 8 External links Couples ( edit ) This section does not cite any sources . Please help improve this section by
adding citations to reliable sources . Unsourced material may be challenged and removed .
</item 4>

<item 5>
Dancing with the Stars (U.S. season 1)
Couples Highest Scoring Dances Lowest Scoring Dances Kelly & Alec Freestyle ( 30 ) Waltz ( 13 ) John & Charlotte Foxtrot , Paso doble , Quickstep & Freestyle (
27 ) Cha - cha - cha ( 20 ) Joey & Ashly Paso doble ( 25 ) Cha - cha - cha , Samba and Foxtrot ( 20 ) Rachel & Jonathan Tango ( 26 ) Waltz ( 20 ) Evander &
Edyta Cha - cha - cha ( 18 ) Jive ( 13 ) Trista & Louis Rumba ( 19 ) Waltz ( 18 ) Weekly scores and songs ( edit ) Unless indicated otherwise , individual
judges scores given in the charts below ( given in parentheses ) are listed in this order from left to right : Carrie Ann Inaba , Len Goodman , Bruno Tonioli .
Week 1 ( edit ) Running order Couple Score Dance Music Joey & Ashly 20 ( 7 , 7 , 6 ) Cha - cha - cha `` Crazy in Love '' -- Beyoncé Rachel & Jonathan 20 ( 7 , 6
,
</item 5>

<item 6>
Dancing with the Stars (U.S. season 1)
They finished third . Scoring chart ( edit ) Couple Place 1 + 2 5 6 Kelly & Alec 13 17 30 21 26 22 + 25 = 47 25 + 30 = 55 John & Charlotte 20 26 46 24 21 27 +
27 = 54 27 + 27 = 54 Joey & Ashly 20 21 41 22 20 20 + 25 = 45 Rachel & Jonathan 20 24 44 26 25 Evander & Edyta 5 18 14 32 13 Trista & Louis 6 18 19 37 indicates
the eliminated couple that week indicates the returning couple that was last to be called safe indicates the winning couple indicates the runner - up couple
indicates the third place couple Red numbers indicate the couples with the lowest score for each week Green numbers indicate the couples with the highest score
for each week Average chart ( edit ) This table only counts for dances scored on a traditional 30 - point scale . Rank by average Place Couple Total Number of
dances Average John & Charlotte 199 8 24.
</item 6>

<item 7>
Dancing with the Stars (U.S. season 1)
Unsourced material may be challenged and removed . ( August 2015 ) ( Learn how and when to remove this template message ) The six professionals and celebrities
that competed were : Celebrity Notability ( known for ) Professional partner Status Trista Sutter Reality television star Louis Van Amstel Eliminated 1st on
June 8 , 2005 Evander Holyfield Professional boxer Edyta Sliwinska Eliminated 2nd on June 15 , 2005 Rachel Hunter Supermodel & actress Jonathan Roberts
Eliminated 3rd on June 22 , 2005 Joey McIntyre New Kids on the Block singer Ashly DelGrosso Third place on June 29 , 2005 John O'Hurley Actor & game show host
Charlotte Jorgensen Runner - up on July 6 , 2005 Kelly Monaco Actress Alec Mazo Winner on July 6 , 2005 † Kelly Monaco returned for the All - Stars season ,
where this time she was paired with Valentin Chmerkovskiy . They finished third .
</item 7>

<item 8>
Dancing with the Stars (U.S. season 1)
Martinez Donald Driver Melissa Rycroft Kellie Pickler Amber Riley Meryl Davis Alfonso Ribeiro Rumer Willis Bindi Irwin Nyle DiMarco Laurie Hernandez Rashad
Jennings Jordan Fisher Professional Alec Mazo Cheryl Burke Julianne Hough Mark Ballas Derek Hough Kym Johnson Karina Smirnoff Peta Murgatroyd Tony Dovolani
Maksim Chmerkovskiy Witney Carson Valentin Chmerkovskiy Emma Slater Lindsay Arnold Spin - off Dancing with the Stars Junior Games Dancing with the Stars We
Dance ! Retrieved from `` https://en.wikipedia.org/w/index.php?title=Dancing_with_the_Stars_(U.S._season_1)&oldid=809378398 '' Categories : Dancing with the
Stars ( U.S. TV series ) 2005 American television seasons Hidden categories : Television season articles that use custom fields Infobox television season
articles that use the season name parameter Articles needing additional references from August 2015 All articles needing additional references Talk Contents
About Wikipedia Español Français Edit links This page was last edited on 8 November 2017 , at 18 : 55 .
</item 8>

<item 9>
Dancing with the Stars (U.S. season 1)
Dancing with the Stars ( U.S. season 1 ) - wikipedia Dancing with the Stars ( U.S. season 1 ) Jump to : navigation , search Dancing with the Stars ( season 1 )
Country of origin United States No. of episodes 8 Release Original network ABC Original release June 1 -- July 6 , 2005 Additional information Celebrity winner
Kelly Monaco Professional winner Alec Mazo Season chronology Next → Season 2 The first season of Dancing with the Stars ( U.S. ) debuted on ABC on June 1 , 2005
. Six celebrities were paired with six professional ballroom dancers . Tom Bergeron and Lisa Canning were the hosts for this season . The judges were Carrie Ann
Inaba , Len Goodman , and Bruno Tonioli . The premiere drew over 13 million viewers , the second biggest summer debut ever for an American reality series after
Survivor . The second week climbed to 15 million viewers , and the show climbed to No. 1 in the summer 2005 TV season ratings , where it remained for the
remainder of the season . The season finale aired July 6 , 2005 .
</item 9>

<item 10>
Dancing with the Stars (U.S. season 1)
6 , 7 ) Waltz `` Three Times A Lady '' -- Lionel Richie Evander & Edyta 18 ( 5 , 7 , 6 ) Cha - cha - cha `` Respect '' -- Aretha Franklin Kelly & Alec 13 ( 5 ,
4 , 4 ) Waltz `` I Have Nothing '' -- Whitney Houston John & Charlotte 20 ( 7 , 7 , 6 ) Cha - cha - cha `` September '' -- Earth , Wind & Fire Trista & Louis 18
( 6 , 6 , 6 ) Waltz `` Come Away with Me '' -- Norah Jones Week 2 ( edit ) Running order Couple Score Dance Music Result Rachel & Jonathan 24 ( 8 , 8 , 8 )
Rumba `` I Do n't Wanna Miss a Thing '' -- Aerosmith from Armageddon Bottom two Joey & Ashly 21 ( 8 , 7 , 6 ) Quickstep `` You 're The One That I Want '' --
John Travolta and Olivia Newton - John Safe Trista & Louis 19 ( 6 , 7 ,
</item 10>
=== END ITEM id=dd6e065e12d75100 ===

=== ITEM id=3bddb639e54f1f5f ===
Gold sentences and facts:
Sentence 0: The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar in 1901 CE , which started as Samvat 1958 . In Nepal , the new year begins with the first day of the month of Baishakh , which usually falls within the months of April -- May in the Gregorian calendar . The first day of the new year is passionately celebrated in a historical carnival that takes place every year in Bhaktapur , called Bisket Jatra.As before , from 2007 AD Nepal Sambat is recognized as the national calender .
  Fact 0.0: The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar.
  Fact 0.1: The Rana rulers made Vikram Samvat the official Hindu calendar in 1901 CE.
  Fact 0.2: Vikram Samvat started as Samvat 1958.
  Fact 0.3: In Nepal the new year begins on the first day of the month of Baishakh.
  Fact 0.4: Baishakh usually falls within April to May in the Gregorian calendar.
  Fact 0.5: The first day of the new year is passionately celebrated in a historical carnival called Bisket Jatra.
  Fact 0.6: The Bisket Jatra carnival takes place every year in Bhaktapur.
  Fact 0.7: As before, from 2007 AD, Nepal Sambat is recognized as the national calendar.
Sentence 1: Vikram Samvat ( Hindi : विक्रम सम्वत् , Nepali : विक्रम सम्वत् ) ( abbreviated as V.S. ( or VS ) or B.S. ( or BS ) ) ; Listen ( help info ) ) is the historical Hindu calendar of India and Nepal . It uses lunar months and solar sidereal year ( see : Vedic time keeping ) . It is used as the official calendar in Nepal .
  Fact 1.0: Vikram Samvat is written in Hindi as विक्रम सम्वत्.
  Fact 1.1: Vikram Samvat is written in Nepali as विक्रम सम्वत्.
  Fact 1.2: Vikram Samvat is abbreviated as V.S. or VS.
  Fact 1.3: Vikram Samvat is abbreviated as B.S. or BS.
  Fact 1.4: Vikram Samvat is the historical Hindu calendar of India and Nepal.
  Fact 1.5: Vikram Samvat uses lunar months.
  Fact 1.6: Vikram Samvat uses the solar sidereal year.
  Fact 1.7: Vikram Samvat is used as the official calendar in Nepal.

Retrieved items:
<item 1>
Vikram Samvat
Vikram Samvat - Wikipedia Vikram Samvat Jump to : navigation , search It has been requested that the title of this article be changed to Bikram Sambat . Please
see the relevant discussion on the discussion page . Do not move the page until the discussion has reached consensus for the change and is closed . It has been
suggested that Bikrami calendar be merged into this article . ( Discuss ) Proposed since October 2017 . Vikram Samvat ( Hindi : विक्रम सम्वत् , Nepali : विक्रम
सम्वत् ) ( abbreviated as V.S. ( or VS ) or B.S. ( or BS ) ) ; Listen ( help info ) ) is the historical Hindu calendar of India and Nepal . It uses lunar months
and solar sidereal year ( see : Vedic time keeping ) . It is used as the official calendar in Nepal . The Vikram Samvat has two alternative systems . It started
in 56 BCE in southern ( purnimanta ) and 57 -- 56 BCE in northern ( amanta ) systems of Hindu calendar .
</item 1>

<item 2>
Vikram Samvat
See also ( edit ) Bikrami calendar Hindu calendars Vira Nirvana Samvat References ( edit ) Jump up ^ Richard Salomon ( 1398 ) . Indian Epigraphy : A Guide to
the Study of Inscriptions in Sanskrit , Prakrit , and the other Indo - Aryan Languages . Oxford University Press . pp. 181 -- 183 . ISBN 978 - 0 - 19 - 535666 -
3 . ^ Jump up to : Ashvini Agrawal ( 1989 ) . Rise and Fall of the Imperial Guptas . Motilal Banarsidass Publ . pp. 174 -- 175 . ISBN 978 - 81 - 208 - 0592 - 7
. Jump up ^ Encyclopedia of New Year 's Holidays Worldwide , William D. Crump , Publisher McFarland , 2014 p. 38 Jump up ^ `` Vikram Samvat should be declared
national calendar '' . The Free Press Journal . 15 February 2012 . Archived from the original on 26 April 2014 . Retrieved 28 March 2012 .
</item 2>

<item 3>
Vikram Samvat
Republican Germanic Greek Hindu Inca Macedonian Maya Haab ' Tzolk'in Muisca Pentecontad Pisan Rapa Nui Roman calendar Rumi Soviet Swedish Turkmen By specialty
Holocene ( anthropological ) Proleptic Gregorian / Proleptic Julian ( historiographical ) Darian ( Martian ) Dreamspell ( New Age ) Discordian / Pataphysical (
surreal ) Proposals Calendar reform Hanke -- Henry Permanent International Fixed Pax Positivist Symmetry454 Tranquility World New Earth Time Fictional Discworld
Greyhawk Middle - earth Stardate Star Wars ( Galactic Standard Calendar ) Displays and applications Electronic Perpetual Wall Year naming and numbering
Terminology Era Epoch Regnal name Regnal year Year zero Systems Ab urbe condita Anno Domini / Common Era Anno Mundi Assyrian Before Present Chinese Imperial
Chinese Minguo Human Era Japanese Korean Seleucid Spanish Yugas Satya Treta Dvapara Kali Vietnamese List of calendars Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Vikram_Samvat&oldid=822253746 '' Categories : Specific calendars Nepali calendar Hindu calendar Memorials to
Vikramaditya Calendar eras
</item 3>

<item 4>
Vikram Samvat
Converting Vikram Samvat to CE ( edit ) While an approximate conversion can be made by subtracting 56 or 57 , a precise computation requires identification of
the version of Vikam Samvat that was used : Year expired ( default ) or current Kartikadi ( older or western India ) or Chaitradi ( modern Northern India ) .
Kartikadi starts 7 months later in Kartika . Until 12th century Kartikadi year was common even in North India . Months Amanta ( south ) or Purminata ( North ) .
Amanta Chaitra Krishna Paksha is Purminata Vaishakh Krishna Paksha . For the Shukla Paksh , the month is the same . Software named Pancanga is available to make
the calculations . Thus the date Vikram 1208 ( Kartikadi , expired ) , Vaishakh Krishna 5 ( Purnimanta ) , Thursday , is equivalent to Vikram 1209 ( Chaitradi ,
expired ) , Chaitra Krishna 5 ( Amanta ) , and to Thursday March 27 , 1152 .
</item 4>

<item 5>
Vikram Samvat
It occurs in mid-April each year , and coincides with the traditional new year in Assam , Bengal , Burma , Cambodia , Kerala , Kashmir , Manipur , Odisha ,
Punjab , Sri Lanka , Tamil Nadu and Thailand . In addition to Nepal , the Vikram Samvat calendar is also recognized in North and East India , and in Gujarat
among Hindus . In Buddhist communities , the month of Baishakh is associated with Vesak or Buddha 's Birthday . It commemorates the birth , Enlightenment and
passing of Gautama Buddha on the first full moon day in May , except in a leap year when the festival is held in June . Although this festival is not held on
the same day as Pahela Baishakh , the holidays typically fall in the same month ( Baishakh ) of the Bengali , Hindu , and Theravada Buddhist calendars , and are
related historically through the spread of Hinduism and Buddhism in South Asia . In Gujarat , the second day of Diwali is celebrated as the first day of the
Vikram Samvat calendar which is the first day of the month Kartik .
</item 5>

<item 6>
Vikram Samvat
According to Rudolf Hoernlé , the king responsible for this change was Yashodharman : Hoernlé also believed that he conquered Kashmir , and is same person as
the `` Harsha Vikramaditya '' mentioned in Kalhana 's Rajatarangini . Earlier , some scholars believed that the Vikrama Samavat corresponded to the Azes era of
the Indo - Scythian ( Śaka ) king King Azes . However , this was disputed by Robert Bracey following discovery of an inscription of Vijayamitra , which is dated
in two eras . The theory seems to be now thoroughly discredited by Falk and Bennett , who place the inception of the Azes era in 47 -- 46 BCE . Culture ( edit )
The traditional New Year of Vikram Samvat is one of the many festivals of Nepal , marked by parties , family gatherings , the exchange of good wishes , and
participation in rituals to ensure good fortune in the coming year .
</item 6>

<item 7>
Vikram Samvat
New International Encyclopedia . 1905 . ( hide ) Calendars Systems Lunar Lunisolar Solar In wide use Astronomical Bengali Chinese Ethiopian Gregorian Hebrew
Hindu Iranian Islamic ISO Unix time In more limited use Akan Armenian Assyrian Bahá'í ( Badí ' ) Balinese pawukon Balinese saka Berber Buddhist Burmese Chinese
Coptic Gaelic Germanic Heathen Georgian Hebrew Hindu or Indian Vikram Samvat Saka Igbo Iranian Jalali ( medieval ) Hijri ( modern ) Zoroastrian Islamic Fasli
Tabular Jain Japanese Javanese Korean Juche Kurdish Lithuanian Malayalam Mongolian Melanau Nanakshahi Nepal Sambat Nisg̱a'a Oromo Romanian Somali Sesotho Slavic
Slavic Native Faith Tamil Thai lunar solar Tibetan Vietnamese Xhosa Yoruba Types Runic Mesoamerican Long Count Calendar round Christian variants Julian Revised
Liturgical year Eastern Orthodox Saints Historical Attic Aztec Tonalpohualli Xiuhpohualli Babylonian Bulgar Byzantine Celtic Cham Culāsakaraj Egyptian
Florentine French Republican Germanic Greek Hindu Inca Macedonian Maya Haab ' Tzolk'in Muisca
</item 7>

<item 8>
Vikram Samvat
377 exactly ) mid-March to mid-April The exact length of each month is the time taken by the Sun to move through a full zodiac sign . History ( edit )
Vikramaditya legends ( edit ) The Jain monk Kalakacharya and the Saka King ( Kalakacharya Katha manuscript , Chhatrapati Shivaji Maharaj Vastu Sangrahalaya ,
Mumbai ) According to popular tradition , the legendary king Vikramaditya of Ujjain established the Vikrama Samvat era after defeating the Śakas . Kalakacharya
Kathanaka ( `` An account of the monk Kalakacharya '' ) by the Jain sage Mahesarasuri gives the following account : Gandharvasena , the then - powerful king of
Ujjain , abducted a nun called Sarasvati , who was the sister of the monk ( citation missing ) . The enraged monk sought the help of the Śaka ruler King Sahi in
Sistan . Despite heavy odds but aided by miracles , the Śaka king defeated Gandharvasena and made him a captive . Sarasvati was repatriated , although
Gandharvasena himself was forgiven .
</item 8>

<item 9>
Vikram Samvat
In India , the reformulated Saka Calendar is officially used ( although not for computing the dates of the traditional festivals ) , although in the Hindi
version of the Preamble of the Constitution of India , the date of adoption of the constitution , 26 November 1949 , is presented in Vikram Samvat ( Margsheersh
Shukla Saptami Samvat 2006 ) . There have been calls for the Vikram Samvat to replace Saka as India 's official calendar . Contents ( hide ) 1 Divisions of a
year 1.1 Lunar metrics 2 History 2.1 Vikramaditya legends 2.2 Historicity of the legends 3 Culture 4 Converting Vikram Samvat to CE 5 See also 6 References 7
Further reading Divisions of a year ( edit ) The classical Vikram Samvat uses lunar months and solar sidereal years . Because 12 months do not match a sidereal
year exactly , correctional months ( adhika māsa ) are added , or occasionally subtracted ( kshaya masa ) .
</item 9>

<item 10>
Vikram Samvat
The Shukla Paksha in both systems coincides , most festivals occur in the Shukla Paksha . The era is named after King Vikramaditya of India . The lunisolar
Vikram Samvat calendar is 56.7 years ahead ( in count ) of the solar Gregorian calendar . For example , the year 2074 VS began in 2017 CE and will end in 2018
CE . The Rana rulers of Nepal made Vikram Samvat the official Hindu calendar in 1901 CE , which started as Samvat 1958 . In Nepal , the new year begins with the
first day of the month of Baishakh , which usually falls within the months of April -- May in the Gregorian calendar . The first day of the new year is
passionately celebrated in a historical carnival that takes place every year in Bhaktapur , called Bisket Jatra.As before , from 2007 AD Nepal Sambat is
recognized as the national calender .
</item 10>
=== END ITEM id=3bddb639e54f1f5f ===

=== ITEM id=a7ae5479195d04bb ===
Gold sentences and facts:
Sentence 0: Originally an ordinary man by the name of Nick Chopper ( the name first appearing in The Marvelous Land of Oz ) , the Tin Woodman used to make his living chopping down trees in the forests of Oz , as his father had before him . The Wicked Witch of the East enchanted his axe to prevent him from marrying his sweetheart , after being bribed by the lazy old woman who kept the Munchkin maiden as a servant , and did not wish to lose her . ( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick 's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no longer able to love the lady he had fallen for .
  Fact 0.0: Nick Chopper was originally an ordinary man.
  Fact 0.1: The name Nick Chopper first appears in The Marvelous Land of Oz.
  Fact 0.2: Nick Chopper used to make his living chopping down trees in the forests of Oz.
  Fact 0.3: Nick Chopper's father had also chopped down trees in the forests of Oz.
  Fact 0.4: The Wicked Witch of the East enchanted Nick Chopper's axe to prevent him from marrying his sweetheart.
  Fact 0.5: The Wicked Witch of the East was bribed to do this by the lazy old woman who kept the Munchkin maiden as a servant.
  Fact 0.6: The lazy old woman did not wish to lose the Munchkin maiden servant.
  Fact 0.7: In a later book of the series, The Tin Woodman of Oz, the woman is said to be the Witch's servant.
  Fact 0.8: In The Tin Woodman of Oz, the Witch herself decides to enchant Nick's axe.
  Fact 0.9: The enchanted axe chopped off Nick Chopper's limbs one by one.
  Fact 0.10: Each time Nick lost a limb, Ku-Klip the tinsmith replaced it with a prosthetic limb made of tin.
  Fact 0.11: Finally, nothing was left of Nick Chopper but tin.
  Fact 0.12: Ku-Klip neglected to replace Nick Chopper's heart.
  Fact 0.13: Once Nick Chopper was made entirely of tin, he was no longer able to love the lady he had fallen for.
Sentence 1: In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest , as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West . The Tin Woodman 's axe proves useful in this journey , both for chopping wood to create a bridge or raft as needed , and for chopping the heads off animals that threaten the party . When the Winged monkeys are sent by the Witch of the West against the group , they throw the Tin Woodman from a great height , damaging him badly . However Winkie Tinsmiths are able to repair him after the death of the Witch .
  Fact 1.0: In The Wonderful Wizard of Oz, Dorothy Gale befriends the Tin Woodman.
  Fact 1.1: Dorothy and her friends find the Tin Woodman rusted in the forest.
  Fact 1.2: The Tin Woodman was caught in rain.
  Fact 1.3: Dorothy uses the Tin Woodman's oil can to release him.
  Fact 1.4: The Tin Woodman follows Dorothy to the Emerald City to get a heart from the Wizard.
  Fact 1.5: The Scarecrow and the Cowardly Lion join Dorothy and the Tin Woodman on their adventure.
  Fact 1.6: The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West.
  Fact 1.7: The Tin Woodman's axe proves useful on this journey for chopping wood to create a bridge or raft as needed.
  Fact 1.8: The Tin Woodman's axe is useful for chopping the heads off animals that threaten the party.
  Fact 1.9: The Winged Monkeys are sent by the Witch of the West against the group.
  Fact 1.10: The Winged Monkeys throw the Tin Woodman from a great height.
  Fact 1.11: The fall damages the Tin Woodman badly.
  Fact 1.12: Winkie Tinsmiths are able to repair the Tin Woodman after the death of the Witch.
Sentence 2: His desire for a heart notably contrasts with the Scarecrow 's desire for brains , reflecting a common debate between the relative importance of the mind and the emotions . This occasions philosophical debate between the two friends as to why their own choices are superior ; neither convinces the other , and Dorothy , listening , is unable to decide which one is right . Symbolically , because they remain with Dorothy throughout her quest , she is provided with both and need not select . The Tin Woodman states unequivocally that he has neither heart nor brain , but cares nothing for the loss of his brain . Towards the end of the novel , though , Glinda praises his brain as not quite that of the Scarecrow 's .
  Fact 2.0: The Tin Woodman's desire for a heart notably contrasts with the Scarecrow's desire for brains.
  Fact 2.1: The contrast reflects a common debate between the relative importance of the mind and the emotions.
  Fact 2.2: The two friends have a philosophical debate about why their own choices are superior.
  Fact 2.3: Neither friend convinces the other in that debate.
  Fact 2.4: Dorothy, listening, is unable to decide which one is right.
  Fact 2.5: Symbolically, because the Tin Woodman and the Scarecrow remain with Dorothy throughout her quest, she is provided with both a heart and a brain and need not select.
  Fact 2.6: The Tin Woodman states unequivocally that he has neither heart nor brain.
  Fact 2.7: The Tin Woodman cares nothing for the loss of his brain.
  Fact 2.8: Towards the end of the novel, Glinda praises the Tin Woodman's brain.
  Fact 2.9: Glinda says the Tin Woodman's brain is not quite that of the Scarecrow.

Retrieved items:
<item 1>
Tin Woodman
When they get to the heart , Zelena picks it up and felt some of her magic draining . When Zelena claims that Dorothy Gale put Stanum up to this task , Stanum
commented that he never met Dorothy . He also stated that Zelena once claimed on the day they met that she would be happy being a normal human . As Stanum fully
turns to tin and is unable to move , Zelena takes the box containing the Crystal Heart and leaves while telling Stanum that perhaps Dorothy will meet Stanum and
free him . Modern works ( edit ) This article gives self - sourcing examples without describing their significance in the context of the article . Please help
improve this article by adding citations to reliable sources that describe the examples ' significance , and by removing less pertinent examples . Unsourced or
poorly sourced material may be challenged or removed . ( November 2017 ) ( Learn how and when to remove this template message ) The song `` Country Robot / A
Letter to Dorothy '' by The Incredible Moses Leroy is written from the Tin Man 's perspective ; it includes the lyrics `` You gave me oil , I was a rusty load /
You even helped me find my heart . ''
</item 1>

<item 2>
Tin Woodman
Toward the end , he hugged the Wizard ( Joel Grey ) saying to him `` Thank you from the bottom of my heart ! '' . In 1996 , they made a cartoon animated The Oz
Kids . He has a son named Tin Boy . Like his father he can cut wood with his ax . And he has to be careful with water and snow or he 'll be rust . His voice
played by Andy Milder . Hiroki Tsujiai from Ultra Maniac dresses as the Tin Man in the costume party . James Kall portrayed the Tin Man in the 2001 ABC
miniseries Life with Judy Garland : Me and My Shadows . The Muppet Gonzo plays a similar role , the Tin Thing , in 2005 's television film The Muppets ' Wizard
of Oz . In this version , he is the Wicked Witch 's research assistant , transformed into a robot to prevent him wanting a day off to marry Camilla . Gonzo 's
other role is himself . He appears at the end of this film in the Muppets ' show .
</item 2>

<item 3>
Tin Woodman
The Tin Woodman 's axe proves useful in this journey , both for chopping wood to create a bridge or raft as needed , and for chopping the heads off animals that
threaten the party . When the Winged monkeys are sent by the Witch of the West against the group , they throw the Tin Woodman from a great height , damaging him
badly . However Winkie Tinsmiths are able to repair him after the death of the Witch . His desire for a heart notably contrasts with the Scarecrow 's desire for
brains , reflecting a common debate between the relative importance of the mind and the emotions . This occasions philosophical debate between the two friends
as to why their own choices are superior ; neither convinces the other , and Dorothy , listening , is unable to decide which one is right . Symbolically ,
because they remain with Dorothy throughout her quest , she is provided with both and need not select . The Tin Woodman states unequivocally that he has neither
heart nor brain , but cares nothing for the loss of his brain . Towards the end of the novel , though , Glinda praises his brain as not quite that of the
Scarecrow 's .
</item 3>

<item 4>
Tin Woodman
The movie was made available for free viewing online and free downloading of its soundtrack . In the 2013 prequel , Oz the Great and Powerful , although the Tin
Woodman does not physically appear , the film introduces his creator , the Master Tinker , portrayed by Bill Cobbs . Tin Man appears in the animated film
Legends of Oz : Dorothy 's Return ( which is based on Dorothy of Oz ) , voiced by Kelsey Grammer . The Woodsman , a 2012 stage play written by James Ortiz with
music by Edward W. Hardy , tells the backstory of the Tin Man , using puppetry , movement , and music . The play has received multiple Off Broadway productions
, and won a 2016 Obie Award for Ortiz 's puppet design . In the TV series Emerald City , the Tin Woodman equivalent is Jack ( Gerran Howell ) , a friend of the
enigmatic Tip , who helps ' him ' escape from his imprisonment by a witch . When it is revealed that Tip is actually a girl whose true identity was suppressed
by a potion , her strained emotional state causes her to push Jack over a balcony after he kisses her .
</item 4>

<item 5>
Tin Woodman
( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick
's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of
tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no
longer able to love the lady he had fallen for . In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest
, as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their
adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West .
</item 5>

<item 6>
Tin Woodman
) The Magical Mimics in Oz ( 1946 ) The Shaggy Man of Oz ( 1949 ) The Hidden Valley of Oz ( 1951 ) Merry Go Round in Oz ( 1963 ) Yankee in Oz ( 1972 ) The
Enchanted Island of Oz ( 1976 ) The Forbidden Fountain of Oz ( 1980 ) The Ozmapolitan of Oz ( 1986 ) The Wicked Witch of Oz ( 1993 ) The Giant Garden of Oz (
1993 ) The Runaway in Oz ( 1995 ) The Rundelstone of Oz ( 2000 ) The Emerald Wand of Oz ( 2005 ) Trouble Under Oz ( 2006 ) Alternate Dorothy of Oz ( 1989 )
Locations and elements Land of Oz Emerald City Munchkin Country Gillikin Country Winkie Country Quadling Country Yellow brick road Oogaboo Deadly Desert
Nonestica Forest of Burzee Land of Ev Merryland Mo Ix Nome Kingdom Noland Ruby slippers Silver Shoes The Oz Film Manufacturing Company Characters Dorothy Gale
Toto Princess Ozma The Wizard of Oz Scarecrow Tin Woodman Cowardly Lion Jack Pumpkinhead Tik - Tok Sawhorse Trot Hungry Tiger
</item 6>

<item 7>
Tin Woodman
They built a story around them , added Dorothy , and added a series of lessons to the effect that everyone possesses the resources they need ( such as brains ,
a heart and courage ) if only they had self - confidence . The Wonderful Wizard of Oz was a children 's book , of course , but as Baum warned in the preface ,
it was a `` modernized '' fairy tale as well . The Tin Man -- the human turned into a machine -- was a common feature in political cartoons and in
advertisements in the 1890s . Indeed , he had been part of European folk art for 300 years . In political interpretations of The Wonderful Wizard of Oz , the
Tin Woodman is supposedly described as a worker , dehumanized by industrialization . The Tin Woodman little by little lost his natural body and had it replaced
by metal ; so he has lost his heart and can not move without the help of farmers ( represented by the Scarecrow ) ; in reality he has a strong sense of
cooperation and love , which needs only an infusion of self - confidence to be awakened . In the 1890s many argued that to secure a political revolution a
coalition of Farmers and Workers was needed .
</item 7>

<item 8>
Tin Woodman
The movie shows him moving through a colorful Oz city with his brand - new mechanical heart , before meeting a reddish female Tin Woman ( or robot ? ) to whom
he gives the heart . She proceeds to cruelly play with the heart . An internet - collaboration , CG animated feature based on Baum 's book The Tin Woodman of Oz
was produced by A : M Films , and completed in 2009 . In 2010 , Whitestone Motion Pictures produced a 23 - minute live - action short film , Heartless : The
Story of the Tin Man . The film is based on the book The Tin Woodman of Oz , where Woot the Wanderer visits the Tin Man and asks how he came to be made of tin .
In the film , the Tin Man appears to be more steam - or coal - powered . His chest cavity is covered by a door which reads `` Pedudoe Tin Co . '' but this is a
reference to the workings of the film company , and not a reference to any Oz book or material . The movie was made available for free viewing online and free
downloading of its soundtrack .
</item 8>

<item 9>
Tin Woodman
The Wizard turns out to be a `` humbug '' and can only provide a placebo heart made of silk and filled with sawdust . However , this is enough to please the Tin
Woodman , who , with or without a heart , was all along the most tender and emotional of Dorothy 's companions ( just as the Scarecrow was the wisest and the
Cowardly Lion the bravest ) . When he accidentally crushes an insect , he is grief - stricken and , ironically , claims that he must be careful about such
things , while those with hearts do not need such care . This tenderness remains with him throughout the series , as in The Patchwork Girl of Oz , where he
refuses to let a butterfly be maimed for the casting of a spell . When Dorothy returns home to her farm in Kansas , the Tin Woodman returns to the Winkie
Country to rule as emperor . Later , he has his subjects construct a palace made entirely of tin -- from the architecture all the way down to the flowers in the
garden . Baum emphasized that the Tin Woodman remains alive , in contrast to the windup mechanical man Tik - Tok Dorothy meets in a later book .
</item 9>

<item 10>
Tin Woodman
To prevent himself from completely becoming tin , Stanum seeks out help from Zelena ( now the Wicked Witch of the West ) at the Emerald City of Oz . Zelena
agrees to help him seek out the Crimson Heart , which can save him . During their quest , Stanum tells Zelena that she does n't have to be lonely but she is
doing her best to deny his advice . Suddenly , a lion comes of nowhere to attack Stanum , and Zelena uses her magic to make the lion go away ( at this point the
lion has become as Zelena would put it , cowardly ) . When they finally arrived to the location of the Crimson Heart , the two learned that the only way to make
it work is through the absorption of another person 's magic . Unfortunately Zelena 's actions and selfish greed for magic causes her to betray Stanum , whom
she suspect was aligned with Dorothy by keeping the Crimson Heart for herself , leaving Stanum to transform into the Tin Man permanently . Some time later ,
when Robin Hood arrives in Oz in the episode `` Heart of Gold '' , the Tin Man is seen on the side of the Yellow Brick Road , torn apart .
</item 10>
=== END ITEM id=a7ae5479195d04bb ===
