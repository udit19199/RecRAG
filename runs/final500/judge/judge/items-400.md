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

This file has 10 items.

=== ITEM id=e1cffe06d89e0136 ===
Gold sentences and facts:
Sentence 0:  After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in four games, and win their second consecutive and second overall Stanley Cup.
  Fact 0.0: The Blackhawks jumped to an early 4–1 lead in the first game of the series.
  Fact 0.1: Mario Lemieux and the Penguins came back to win the first game of the series.
  Fact 0.2: Mario Lemieux and the Penguins swept the series in four games.
  Fact 0.3: The Penguins won their second consecutive Stanley Cup.
  Fact 0.4: The Penguins won their second overall Stanley Cup.
Sentence 1: Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.
  Fact 1.0: Mario Lemieux was born on October 5, 1965.
  Fact 1.1: Mario Lemieux has the honorifics OC and CQ.
  Fact 1.2: Mario Lemieux is a Canadian.
  Fact 1.3: Mario Lemieux is a former professional ice hockey player.
  Fact 1.4: Mario Lemieux is the current owner of the Pittsburgh Penguins team.

Retrieved items:
<item 1>
History of the National Hockey League
The history of the National Hockey League begins with the end of its predecessor league, the National Hockey Association (NHA), in 1917.

 After unsuccessfully attempting to resolve disputes with Eddie Livingstone, owner of the Toronto Blueshirts, executives of the three other NHA franchises
suspended the NHA, and formed the National Hockey League (NHL), replacing the Livingstone team with a temporary team in Toronto, the Arenas.

 The NHL's first quarter-century saw the league compete against two rival major leagues—the Pacific Coast Hockey Association and Western Canada Hockey
League—for players and the Stanley Cup.

 The NHL first expanded into the United States in 1924 with the founding of the Boston Bruins, and by 1926 consisted of ten teams in Ontario, Quebec, the Great
Lakes region, and the Northeastern United States.

 At the same time, the NHL emerged as the only major league and the sole competitor for the Stanley Cup; in 1947, the NHL completed a deal with the Stanley Cup
trustees to gain full control of the Cup.

 The NHL's footprint spread across Canada as Foster Hewitt's radio broadcasts were heard coast-to-coast starting in 1933.
</item 1>

<item 2>
1917–18 Toronto Hockey Club season
The 1917–18 Toronto Hockey Club season was the first season of the new "Toronto" franchise in the newly organized National Hockey League (NHL).

 The team was intended as a 'temporary' franchise, operating without an official club nickname (the press would dub them the "Blue Shirts" or "Torontos", and in
1948 the NHL would engrave "Toronto Arenas" on the Stanley Cup as the 1917–18 winner) and without a formal organization separate from the Toronto Arena Company
that managed the Arena Gardens.

 Despite this, the team came together to win the first NHL Championship, competing against existing teams that had transferred directly from the National Hockey
Association (NHA).

 Toronto would go on to win the Stanley Cup by defeating the Pacific Coast Hockey Association champion Vancouver Millionaires – the first Stanley Cup for an NHL
team and the second Cup for a Toronto team after the Toronto Blueshirts' victory in the 1913–14 season of the NHA.
</item 2>

<item 3>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 3>

<item 4>
1983 Stanley Cup Finals
The 1983 Stanley Cup Finals was contested by the Edmonton Oilers in their first-ever Finals appearance and the defending champion New York Islanders, in their
fourth, and fourth consecutive, Finals appearance.

 The Islanders would win the best-of-seven series four games to none, to win their fourth-straight and fourth-overall Stanley Cup.

 It was also the fourth straight Finals of post-1967 expansion teams, and the first involving a former World Hockey Association (WHA) team.

 This is also the most recent time that a defending Stanley Cup champion has won the cup four years in a row, and also the first (and, to date, only) time a
North American professional sports team has won four consecutive titles in any league competition with more than twenty teams.

 Since 1983, no professional sports team on the continent has managed to win four straight championships and no NHL team has won more than two consecutive
championships (most recently the Pittsburgh Penguins in and ).
</item 4>

<item 5>
1998 Stanley Cup Finals
The 1998 Stanley Cup Finals was the championship series of the National Hockey League's (NHL) 1997–1998 season, and the culmination of the 1998 Stanley Cup
playoffs.

 It was contested by the Western Conference champion and defending Stanley Cup champion Detroit Red Wings and the Eastern Conference champion Washington
Capitals.

 It was the 105th year of the Stanley Cup being contested.

 The series was the Capitals' first appearance in a Stanley Cup Final since the franchise's inception in 1974.

 The Red Wings won the series for the second year in a row, four games to none.

 It was the Wings' ninth Stanley Cup, and the most recent time when a Finals concluded with a sweep (as of 2017).

 This was also the last time until that a Stanley Cup Finals ended after an NBA Finals in the same season had concluded.
</item 5>

<item 6>
Mike Babcock
Michael "Mike" Babcock Jr. (born April 29, 1963) is a Canadian professional ice hockey head coach, currently serving as head coach of the Toronto Maple Leafs of
the National Hockey League (NHL).

 He previously spent 10 seasons as head coach of the Detroit Red Wings, winning the Stanley Cup with them in 2008 and helping them to the Stanley Cup Playoffs
every year during his tenure.

 He holds the record as the coach with the most wins in Red Wings history, surpassing that of Jack Adams.

 He has also served as head coach of the Mighty Ducks of Anaheim, whom he helped to the 2003 Stanley Cup Finals.
</item 6>

<item 7>
1998 Stanley Cup Finals
Detroit coach Scotty Bowman won his eighth Stanley Cup in that capacity (having previously done so with the Montreal Canadiens in , , , , and , the Pittsburgh
Penguins in , and the Wings the previous year), tying him with former Canadiens coach Toe Blake for the record of most Cups won by a coach (which he would break
when he helped the Red Wings win the 2002 Cup).
</item 7>

<item 8>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 8>

<item 9>
Mario Lemieux
Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.

 He played parts of 17 seasons in the National Hockey League's (NHL) with the Pittsburgh Penguins, between 1984 and 2006.

 Dubbed "The Magnificent One" or "Le Magnifique" (as well as "Super Mario"), he is widely acknowledged to have been one of the best players of all time.

 He is the only player to score one goal in each of the five possible situations in a single NHL game, a feat he accomplished in 1988.

 A gifted playmaker and fast skater despite his large size, Lemieux often beat defencemen with fakes and dekes.
</item 9>

<item 10>
1984 Stanley Cup Finals
The 1984 Stanley Cup Final was held between the Edmonton Oilers and the then-defending champion New York Islanders.

 The Islanders had swept the Oilers in four straight games to win the Cup.

 In 1984, the Islanders were seeking their fifth consecutive Stanley Cup championship, but the upstart Oilers would win the best-of-seven series four games to
one to win their first Stanley Cup, becoming the third post-1967 expansion team and first former World Hockey Association team to win the Cup, and also the
first team based west of Chicago to win the Cup since the WCHL's Victoria Cougars became the last non-NHL team to win it in .

 It was also the fifth straight Finals of teams that joined the NHL in 1967 or later and a rematch of the 1983 Finals—a Stanley Cup Finals rematch would not
happen again until the Finals.

 s of 2017 , the Islanders' four consecutive Cup wins (, , , 1983) and their appearance in the 1984 Cup Finals is an NHL record of 19 consecutive playoff series
wins that currently stands unbroken.
</item 10>
=== END ITEM id=e1cffe06d89e0136 ===

=== ITEM id=8576c17c3f134519 ===
Gold sentences and facts:
Sentence 0: Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 0.0: Samsung Galaxy S7 is an Android smartphone.
  Fact 0.1: Samsung Galaxy S7 Edge is an Android smartphone.
  Fact 0.2: Samsung Galaxy S7 and S7 Edge are manufactured by Samsung Electronics.
  Fact 0.3: Samsung Galaxy S7 and S7 Edge are marketed by Samsung Electronics.
Sentence 1:  The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.
  Fact 1.0: The S7 series is the successor to the Galaxy S6.
  Fact 1.1: The S7 series is the successor to the Galaxy S6 Edge.
  Fact 1.2: The S7 series is the successor to the Galaxy S6 Edge+.
  Fact 1.3: The Galaxy S6, S6 Edge and S6 Edge+ were released in 2015.
Sentence 2: Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 2.0: Samsung Galaxy S6 is an Android smartphone.
  Fact 2.1: Samsung Galaxy S6 Edge is an Android smartphone.
  Fact 2.2: Samsung Galaxy S6 Edge+ is an Android smartphone.
  Fact 2.3: Samsung Galaxy S6, S6 Edge and S6 Edge+ are manufactured by Samsung Electronics.
  Fact 2.4: Samsung Galaxy S6, S6 Edge and S6 Edge+ are marketed by Samsung Electronics.
Sentence 3:  The S6 line serves as a successor to the Galaxy S5.
  Fact 3.0: The S6 line is a successor to the Galaxy S5.
Sentence 4:  The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.
  Fact 4.0: The S6 was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.1: The S6 Edge was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.2: The first Samsung Unpacked 2015 event was held at Mobile World Congress on 1 March 2015.
  Fact 4.3: The S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5.
  Fact 4.4: The S6 Edge+ was officially unveiled in the second Samsung Unpacked 2015 event.
  Fact 4.5: The second Samsung Unpacked 2015 event was held in New York on 13 August 2015.

Retrieved items:
<item 1>
Samsung SGH-E900
The Samsung E900, introduced in 2006, is a high-end mobile phone and is derived from Samsung's D500-D600-D800 series of slide phones.

 It is of a slide-up design and has touch-sensitive keys, similar to the LG Chocolate (KG800).

 It was one of the most popular phones at the time but was discontinued late 2007 when buyers then turned their attention to the Samsung D900i, followed by the
Samsung U600, which was the slimmest phone by Samsung with a thickness of 1.09 cm until the E840 was released.
</item 1>

<item 2>
Samsung Galaxy Gear
The Samsung Galaxy Gear is a smartwatch produced by Samsung Electronics in the Samsung Gear family of devices.

 Unveiled during a "Samsung Unpacked" event in Berlin, Germany on September 4, 2013, the device serves as a companion for all Samsung Galaxy smartphones and
tablets which run Android 4.3 or newer; it was released on September 25, 2013.

 Originally released as an Android-based device, Samsung replaced the operating system with Tizen through the May 2014 software update.
</item 2>

<item 3>
Samsung Galaxy Tab 8.9
The Samsung Galaxy Tab 8.9 is an Android-based tablet computer designed and manufactured by Samsung, introduced on 22 March 2011 at CTIA wireless convention in
its Samsung Unpacked event in Orlando.

 It is part of the Samsung Galaxy Tab series, and features an 8.9-inch display and a 1 GHz dual-core Nvidia Tegra 2 processor.
</item 3>

<item 4>
Samsung REX
The Samsung REX series is a range of Samsung smartphones and feature phones, ranging from low to mid-end phones, running a Java-based operating system with
Samsung's TouchWiz user interface.

 It is currently one of the Samsung line of devices that are aimed at the budget phone market alongside lower end Samsung Galaxy devices.

 Its main competition includes the Nokia Asha range and Firefox OS.
</item 4>

<item 5>
Samsung SGH-D500
The Samsung SGH-D500 is a slider-style mobile phone created by Samsung.

 It was announced in Q4 2004 as a replacement to the popular Samsung E800.

 It has a 1.3 megapixel digital camera with an integrated LED flash.

 The phone can be purchased in various colors such as black, grey/silver, silver/white, silver/blue, black/blue, and black/silver.

 The D500 won the prestigious award "the worlds best cellphone" at the 3GSM world congress in Cannes, 2005.

 The SGH-D500 was popular with consumers because of the large screen-size to front-surface ratio, the smooth button and sliding action, powerful flash, and good
video/photograph/sound quality.

 The D500 was regarded as compact and as having a good feature set for its time.

 Samsung experienced remarkable success with the introduction of the D500 and has therefore introduced several other slider-style mobile phones in recent years,
some fairly successful, others not so much.

 This phone helped to popularise the 'active' sliding phone concept across all brands.

 The SGH-D500 has been succeeded by the Samsung SGH-D600.
</item 5>

<item 6>
Samsung Galaxy S Duos 2
Samsung Galaxy S Duos 2 is a version of Samsung Galaxy Trend Plus with dual SIM Android smartphone, produced and marketed by Samsung Electronics, which serves
as an immediate successor to the original Galaxy S Duos, which is made in second quarter of 2012.

 Unveiled on 30 November 2013 at some Asian markets, it was initially released on December 5, 2013 in several countries.

 In contrast with other dual sim Samsung models, this phone is a part of the high-end "S" series, this is why it is marketed as a part of the "Galaxy S" family.

 Despite being called as a direct successor to the original S Duos phone, external and physical design of the model is identical to the original model, placing
emphasis on internal upgrades such as upgraded processor and updated Operating System software.
</item 6>

<item 7>
Samsung SGH-T919
The Samsung Behold SGH-T919 is a touch-screen, 3G candybar-style mobile phone introduced by Samsung late in 2008.

 The Behold is one of the first Samsung mobile phones released to have a touch-screen along with the Samsung Omnia, the Samsung Instinct, the Samsung Eternity,
and the Samsung Tocco.

 The Behold is the "American" release of Samsung Tocco, which is only sold in Europe, Asia and Australia so the features and design of both of these phones are
similar.

 Some of the Behold's biggest competitors were Apple's iPhone, T-Mobile G1 (HTC Dream), LG Dare, Samsung Instinct, and Samsung Eternity.

 It was only available only to customers of T-Mobile.
</item 7>

<item 8>
Samsung Galaxy S7
Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.

 The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.

 The phones were officially unveiled on 21 February 2016, during a Samsung press conference at Mobile World Congress, with a European and North American release
on 11 March 2016.
</item 8>

<item 9>
Samsung Ativ S
The Samsung ATIV S is a touchscreen, slate smartphone manufactured by Samsung Electronics running the Windows Phone 8 operating system, upgradeable to Windows
Phone 8.1.

 The ATIV S is Samsung's first Windows Phone 8 device, and one of the first devices under its ATIV series of Microsoft Windows-based products.

 It was shown at the IFA 2012 in Berlin and was the first Windows Phone 8 device to be officially unveiled, ahead of the Nokia Lumia 920.
</item 9>

<item 10>
Samsung Galaxy S6
Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.

 The S6 line serves as a successor to the Galaxy S5.

 The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger
S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.

 Alongside the S6, Samsung also unveiled the S6 Edge (and later on the bigger S6 Edge+), a variant whose screen is wrapped along the sides of the device; the
curvature is usable for several additional features.

 The Galaxy S6 and S6 Edge were released on 10 April 2015 in 20 countries while the S6 Edge+ was released on 21 August 2015 in 20 countries.
</item 10>
=== END ITEM id=8576c17c3f134519 ===

=== ITEM id=7f009948369db296 ===
Gold sentences and facts:
Sentence 0:  Game included wild boar, and red, fallow and roe deer.
  Fact 0.0: Game included wild boar.
  Fact 0.1: Game included red deer.
  Fact 0.2: Game included fallow deer.
  Fact 0.3: Game included roe deer.
Sentence 1:  This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.
  Fact 1.0: The common species is native to western Eurasia.
  Fact 1.1: The common species has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.

Retrieved items:
<item 1>
Fallow deer
The fallow deer ("Dama dama") is a ruminant mammal belonging to the family Cervidae.

 This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar,
Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the
United States, the Falkland Islands, and Peru.

 Some taxonomers include the rarer Persian fallow deer as a subspecies ("D. d. mesopotamica"), while others treat it as an entirely different species ("D.
mesopotamica").
</item 1>

<item 2>
Moose
The moose (North America) or elk (Eurasia), "Alces alces", is the largest extant species in the deer family.

 Moose are distinguished by the broad, flat (or "palmate") antlers of the males; other members of the family have antlers with a dendritic ("twig-like")
configuration.

 Moose typically inhabit boreal forests and temperate broadleaf and mixed forests of the Northern Hemisphere in temperate to subarctic climates. Hunting and
other human activities have caused a reduction in the size of the moose's range over time.

 Moose have been reintroduced to some of their former habitats.

 Currently, most moose are found in Canada, Alaska, New England, Fennoscandia, Baltic States, and Russia.

 Their diet consists of both terrestrial and aquatic vegetation.

 The most common moose predators are the gray wolf along with bears and humans. Unlike most other deer species, moose are solitary animals and do not form
herds.

 Although generally slow-moving and sedentary, moose can become aggressive and move quickly if angered or startled.

 Their mating season in the autumn features energetic fights between males competing for a female.
</item 2>

<item 3>
Red deer
The red deer ("Cervus elaphus") is one of the largest deer species.

 The red deer inhabits most of Europe, the Caucasus Mountains region, Asia Minor, Iran, parts of western Asia, and central Asia.

 It also inhabits the Atlas Mountains region between Morocco and Tunisia in northwestern Africa, being the only species of deer to inhabit Africa.

 Red deer have been introduced to other areas, including Australia, New Zealand, United States, Canada, Peru, Uruguay, Chile and Argentina.

 In many parts of the world, the meat (venison) from red deer is used as a food source.
</item 3>

<item 4>
An encyclopedia-style collection covers deer and other wildlife, noting habitats, ranges, traits, and conservation status: Sri Lankan sambar are exceptionally
large; moose are solitary northern browsers; several Asian deer are threatened, while Indian muntjac is least concern. It also mentions the Calypso
fish-identification database, wild boar, and black bull. Other entries describe the Umba River and Babati; Ovambo, Oku, and Ewe peoples; Hawaiian muumuus
</item 4>

<item 5>
Forests of Mara and Mondrem
The Forests of Mara and Mondrem were adjacent medieval forests in Cheshire, England, which in the 11th century extended to over 60 sqmi , stretching from the
Mersey in the north almost to Nantwich in the south, and from the Gowy in the west to the Weaver in the east.

 Mara and Mondrem were a hunting forest of the Norman Earls of Chester, established soon after 1071 by the first earl, Hugh d'Avranches.

 They might earlier have been an Anglo-Saxon hunting forest.

 Game included wild boar, and red, fallow and roe deer.
</item 5>

<item 6>
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 6>

<item 7>
Indian muntjac
The Indian muntjac ("Muntiacus muntjak"), also called red muntjac and barking deer, is a common muntjac deer species in South and Southeast Asia.

 It is listed as Least Concern on the IUCN Red List.
</item 7>

<item 8>
Sri Lankan sambar deer
The Sri Lankan sambar deer ("Rusa unicolor unicolor") is a subspecies of sambar deer that lives in Sri Lanka.

 This subspecies is one of the largest sambar deer species with the largest antlers both in size and in body proportions.

 Large males weight up to 270–280 kg.

 Sri Lankan sambar live in lowland dry forests and mountain forests.

 Large herds of sambar deer roam the Horton Plains National Park, where it is the most common large mammal.
</item 8>

<item 9>
The text profiles deer, other animals, and a fish database. Deer entries cover habitats, ranges, traits, and conservation: Sri Lankan sambar are exceptionally
large; fallow and red deer have broad native or introduced ranges; moose are solitary, antlered northern browsers; Himalayan musk, Visayan spotted, and marsh
deer are endangered or vulnerable; Indian muntjac is least concern. It also describes the Calypso fish-identification database, wild boar, black bull
</item 9>

<item 10>
Visayan spotted deer
The Visayan spotted deer ("Rusa alfredi"), also known as the Philippine spotted deer, is a nocturnal and endangered species of deer located primarily in the
rainforests of the Visayan islands of Panay and Negros though it once roamed other islands such as Cebu, Guimaras, Leyte, Masbate, and Samar.

 It is one of three endemic deer species in the Philippines, although it was not recognized as a separate species until 1983.

 An estimated 2,500 mature individuals survived worldwide as of 1996, according to the IUCN, although it is uncertain of how many of them still survive in the
wild.

 The diet of the deer, which consists of a variety of different types of grasses, leaves, and buds within the forest, is the primary indicator of its habitat.

 Since 1991 the range of the species has severely decreased and is now almost co-extensive with that of the Visayan warty pig.
</item 10>
=== END ITEM id=7f009948369db296 ===

=== ITEM id=b95b45f47f7d7ff0 ===
Gold sentences and facts:
Sentence 0:  It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black matipo", "Pittosporum tenuifolium", is not related to "Myrsine").
  Fact 0.0: Myrsine contains about 200 species.
  Fact 0.1: Myrsine includes several notable radiations.
  Fact 0.2: The matipo of New Zealand is one of the notable radiations within Myrsine.
  Fact 0.3: The kōlea of Hawaiʻi is one of the notable radiations within Myrsine.
  Fact 0.4: The New Zealand 'black matipo', Pittosporum tenuifolium, is not related to Myrsine.
Sentence 1: Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.
  Fact 1.0: Storax or snowbell are the common names of Styrax.
  Fact 1.1: Styrax is a genus of about 130 species.
  Fact 1.2: Styrax species are large shrubs or small trees.
  Fact 1.3: Styrax belongs to the family Styracaceae.
  Fact 1.4: Styrax is mostly native to warm temperate to tropical regions of the Northern Hemisphere.
  Fact 1.5: The majority of Styrax species are in eastern and southeastern Asia.
  Fact 1.6: Styrax also crosses the equator in South America.

Retrieved items:
<item 1>
Styrax
Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm
temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.

 The resin obtained from the tree is called storax or benzoin (not to be confused with the "Liquidambar" storax balsam).
</item 1>

<item 2>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 2>

<item 3>
Styrax fraserensis
Styrax fraserensis is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is a tree endemic to Peninsular Malaysia.

 It is threatened by habitat loss.
</item 3>

<item 4>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 4>

<item 5>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 5>

<item 6>
Styrax vilcabambae
Styrax vilcabambae (syn.

 "Pamphilia vilcabambae" D.R.Simpson) is a species of flowering plant in the genus "Styrax" and the family Styracaceae.

 It is endemic to Peru.
</item 6>

<item 7>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 7>

<item 8>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 8>

<item 9>
The text covers several flowering plants. Pterostyrax is a four-species East Asian genus of deciduous shrubs and small trees; P. hispidus grows to 15 m and
bears fragrant white summer flowers. Styrax comprises about 130 shrubs and trees, mostly from Asia; its resin, benzoin, is used in perfume, incense, flavoring,
and medicine. Listed Styrax species are native to Malaysia, Singapore, Vietnam, and Peru;
</item 9>

<item 10>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 10>
=== END ITEM id=b95b45f47f7d7ff0 ===

=== ITEM id=bc84c40beb6dd411 ===
Gold sentences and facts:
Sentence 0: Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).
  Fact 0.0: Groom Lake is a salt flat in Nevada.
  Fact 0.1: Groom Lake is used for runways of the Nellis Bombing Range Test Site airport.
  Fact 0.2: The Nellis Bombing Range Test Site airport has the code KXTA.
Sentence 1:  Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
  Fact 1.0: Groom Lake is located within the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.1: Groom Lake Valley is the namesake of the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.2: Groom Lake is 25 miles south of Rachel, Nevada.
Sentence 2: Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.
  Fact 2.0: Rachel is a census-designated place (CDP).
  Fact 2.1: Rachel is in Lincoln County, Nevada, United States.

Retrieved items:
<item 1>
Tonopah Test Range
Tonopah Test Range (TTR), is a restricted military installation located about 30 mi southeast of Tonopah, Nevada.

 It is part of the northern fringe of the Nellis Range, measuring 625 sqmi .

 Tonopah Test Range is located about 70 mi northwest of Groom Dry Lake, the home of the Area 51 facility.

 Like the Groom Lake facility, Tonopah is a site of interest to conspiracy theorists, mostly for its use of experimental and classified aircraft.

 As such, it is not the focus of alien enthusiasts, unlike its neighbor.

 It is currently used for nuclear weapons stockpile reliability testing, research and development of fusing and firing systems, and testing nuclear weapon
delivery systems.

 The airspace comprises restricted area R-4809 of the Nevada Test and Training Range and is often used for military training.
</item 1>

<item 2>
John Skillern House
The John Skillern House is a historic cabin located 25 mi northwest of Fairfield in Camas County, Idaho, near the confluence of the Big Smokey and Little Smokey
creeks.

 The cabin was built in 1921-22 for John Skillern and his wife, who used it as a summer home and headquarters for John's large sheep ranching business.

 Skillern's wife based the cabin's rustic design off of the Old Faithful Inn in Yellowstone National Park, which is reflected in the cabin's steep roof and
covered front porch.

 The cabin's other significant rustic features include its sleeping lofts with pole railings, its horizontal log construction with exposed logs on the inner
walls, and its stone chimney.
</item 2>

<item 3>
Groom Range
The Groom Range is a mountain range in Lincoln County, Nevada.

 It is located within the Nevada Test and Training Range, north of Groom Dry Lake.

 The highest point in the Groom Range is 9,249 feet.

 The Groom Range is situated 26.6 miles north of the dry Groom Lake.
</item 3>

<item 4>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 4>

<item 5>
Groom Lake (salt flat)
Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).

 It is located immediately north of the Area 51 USAF installation.

 The lake at 4409 ft elevation is approximately 3.7 mi from north to south and 3 mi from east to west at its widest point, and is approximately 11.3 miles in
circumference.

 Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
</item 5>

<item 6>
Papoose Lake
Papoose Lake is a dry lake bed located in Lincoln County, Nevada, USA.

 The lake lies within the plot of land referred to as the Groom Lake facility (aka Area 51) and is a restricted area.

 The lake is located a few miles southwest of the Groom lake facility nestled adjacent the Papoose Range.
</item 6>

<item 7>
Leviathan Mine
Leviathan Mine is a United States superfund site (CERCLIS ID: CAD98067685) at an abandoned open-pit sulfur mine located in Alpine County, California.

 The mine is located at on the eastern slope of the Sierra Nevada at about 7,000 ft elevation, 6 mi east of Markleeville and 24 mi southeast of Lake Tahoe.

 The mine site comprises approximately 250 acre of land surrounded by the Toiyabe National Forest, which is only accessible a few months a year.

 The approximately 22 million tons of sulfur ore-containing crushed rock at the mine are responsible for contaminating the Leviathan and Aspen Creek, which join
with Mountaineer Creek to form Bryant Creek which ultimately empties into the East Fork of the Carson River.

 These water bodies are listed as 303(d) impaired.

 The site location is seismically active.
</item 7>

<item 8>
Rachel, Nevada
Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.

 As of the 2010 census it had a population of 54.

 As the closest habitation to the Nellis Air Force Range and Area 51, Rachel enjoys a modest celebrity, particularly among aviation enthusiasts and UFO hunters.

 North of the town is the Quinn Canyon Range, which has the ghost town of Adaven.
</item 8>

<item 9>
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 9>

<item 10>
Squire Creek Country Club
The Squire Creek Country Club is a private, members-only country club located in Choudrant, Louisiana, five miles northeast of Ruston.

 Squire Creek features an 18-hole championship golf course designed by Tom Fazio.

 The course has been ranked as number one in the state of Louisiana four times, and was ranked as the No. 5 best new course in the United States by "Golf
Digest".

 Squire Creek is the home golf course for the Louisiana Tech Bulldogs golf team.

 Squire Creek offers a golf training facility, which features dual bays with retractable doors for indoor or outdoor use, motion analysis, and launch monitor.

 The Squire Creek tennis facility has 6 lighted courts (4 Hydro Courts and 2 Hard Courts).

 Squire Creek has full service golf and tennis shops.

 The Squire Creek fitness facility has 2 exercise rooms and offers massage therapy.

 The Squire Creek Clubhouse features three dining areas: the more formal Main Dining Hall, the casual 19th Hole, and the Fazio Grill.

 The Squire Creek Lodge features two floors offering a combined total of seven bedrooms.
</item 10>
=== END ITEM id=bc84c40beb6dd411 ===

=== ITEM id=98ee5233b978a383 ===
Gold sentences and facts:
Sentence 0: Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.
  Fact 0.0: The Dogo Cubano is also known as the Cuban Mastiff.
  Fact 0.1: The Dogo Cubano is also known as the Cuban Dogo.
  Fact 0.2: The Dogo Cubano is also known as the Cuban Dogge.
  Fact 0.3: The Dogo Cubano is an extinct dog breed.
  Fact 0.4: The Dogo Cubano is a dog breed from Cuba.
Sentence 1: The Dutch Shepherd is a herding dog of Dutch origin.
  Fact 1.0: The Dutch Shepherd is a herding dog.
  Fact 1.1: The Dutch Shepherd is a herding dog of Dutch origin.

Retrieved items:
<item 1>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 1>

<item 2>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 2>

<item 3>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 3>

<item 4>
Podenco Canario
Podenco Canario (In English: "Canary Islands Hound", "Canarian Warren Hound") is a breed of dog originally from the Canary Islands.

 The Podenco Canario is still used in the Canary Islands today in packs as a hunting dog, primarily used for rabbit hunting.

 The word "podenco" is Spanish for "hound".

 "Canario" means "Canarian" or "of the Canaries".
</item 4>

<item 5>
Dogo Cubano
Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.

 It was of the Bull Mastiff type and was used for dog fighting.
</item 5>

<item 6>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 6>

<item 7>
Cuban Criollo
The Criollo Cubano originates from Spanish horses brought to Cuba by Diego Velázquez in 1751.

 There are four different breeds known as criollo: the Cubano de Paso, the Pinto Cubano, the Criollo de Trote and the Patibarcino.

 It is a small stout breed used by the Guajiro people.
</item 7>

<item 8>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 8>

<item 9>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 9>

<item 10>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 10>
=== END ITEM id=98ee5233b978a383 ===

=== ITEM id=28875201cde0bdb0 ===
Gold sentences and facts:
Sentence 0: The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna (Tenerife, Canary Islands, Spain).
  Fact 0.0: The Church of the Guanche People's Spanish name is 'Iglesia del Pueblo Guanche'.
  Fact 0.1: The Church of the Guanche People is a religious organisation.
  Fact 0.2: The Church of the Guanche People was founded in 2001.
  Fact 0.3: The Church of the Guanche People was founded in the city of San Cristobal de La Laguna.
  Fact 0.4: San Cristobal de La Laguna is in Tenerife, Canary Islands, Spain.
Sentence 1:  Tenerife is the largest and most populous island of Macaronesia.
  Fact 1.0: Tenerife is the largest island of Macaronesia.
  Fact 1.1: Tenerife is the most populous island of Macaronesia.

Retrieved items:
<item 1>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 1>

<item 2>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 2>

<item 3>
Garajonay National Park, on La Gomera in Spain, covers 40 km², spans six municipalities, and became a national park in 1981 and UNESCO World Heritage Site in
1986. Bulgaria’s Pirin National Park covers 403.56 km², reaches 2,914 m at Vihren, and has been protected since 1962 and UNESCO-listed since 1983. Spain’s
Guadarrama National Park protects nearly 34,000
</item 3>

<item 4>
Mactan, Philippines, is a densely populated 65-km² island of about 470,000, divided between Lapu-Lapu City and Cordova and linked to Cebu by two bridges.
Norway’s Bjørnøya is mainland-connected by causeway; Loppa’s 1953 church seats 150. Estonia’s Vilsandi is a 9-km² bird refuge and national-park island. Other
entries describe Nova Scotia’s Catcha Lake;
</item 4>

<item 5>
Hermigua, on northeastern La Gomera, lies 12 km from the island capital; Garajonay National Park covers its south. Tenerife, the Canaries’ largest and most
populous island, contains Teide National Park and the emblematic Roque Cinchado. Guanche mummies preserve pre-Spanish indigenous remains; a Guanche pagan
organization was founded in La Laguna in 2001. Chile’s Cay is an apparently inactive stratovolcano; Penco
</item 5>

<item 6>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 6>

<item 7>
Mactan, a densely populated Cebu island, has 470,000 residents. Sino-Tibetan is a 400-plus-language family. Sikh culture intertwines faith and Punjabi heritage.
Amka is an Israeli moshav founded by Yemenite Jews. Endangered Costa Rican Maléku and Cabécar contrast with widely spoken Paraguayan Guarani.
Indigenous-language initiatives include Australia’s Ngapartji Ngapartji project. Haisla live in British
</item 7>

<item 8>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 8>

<item 9>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 9>

<item 10>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 10>
=== END ITEM id=28875201cde0bdb0 ===

=== ITEM id=078e7c967b39df11 ===
Gold sentences and facts:
Sentence 0: Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.
  Fact 0.0: Awolnation is an American alternative rock band.
  Fact 0.1: Awolnation was formed by Aaron Bruno.
  Fact 0.2: Awolnation is fronted by Aaron Bruno.
  Fact 0.3: Aaron Bruno was formerly of Under the Influence of Giants.
  Fact 0.4: Aaron Bruno was formerly of Home Town Hero.
  Fact 0.5: Aaron Bruno was formerly of Insurgence.
Sentence 1: This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.
  Fact 1.0: This Mortal Coil was a music collective.
  Fact 1.1: This Mortal Coil was led by Ivo Watts-Russell.
  Fact 1.2: Ivo Watts-Russell founded the British record label 4AD.

Retrieved items:
<item 1>
Awolnation
Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.
</item 1>

<item 2>
Awolnation discography
American alternative rock band Awolnation has released two studio albums, three extended plays, eight singles and ten music videos.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.

 The band's second studio album, "Run", was released on March 17, 2015.
</item 2>

<item 3>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 3>

<item 4>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 4>

<item 5>
Kill Your Heroes
"Kill Your Heroes" is a song by American alternative rock band Awolnation.

 It is written by lead singer Aaron Bruno and musician Brian West for the band's debut studio album "Megalithic Symphony", where it appears as the eighth track.

 "Kill Your Heroes" was released as the third and final single from "Megalithic Symphony" and reached the top 20 of the United States "Billboard" Alternative
Songs and Rock Songs charts.
</item 5>

<item 6>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 6>

<item 7>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 7>

<item 8>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 8>

<item 9>
Hollow Moon (Bad Wolf)
"Hollow Moon (Bad Wolf)" is a song by American alternative rock band Awolnation.

 It was released as the lead single from their second studio album "Run" (2015).

 The song became their first number-one hit on the Billboard Alternative Songs chart in early 2015.
</item 9>

<item 10>
This Mortal Coil
This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.

 Although Watts-Russell and John Fryer were technically the only two official members, the band's recorded output featured a large rotating cast of supporting
artists, many of whom were otherwise associated with 4AD, including members of Cocteau Twins, Colourbox, and Dead Can Dance.

 The project became known for its gothic, dream pop sound, and released three full albums beginning in 1984 with "It'll End in Tears".
</item 10>
=== END ITEM id=078e7c967b39df11 ===

=== ITEM id=c9738815aed0985a ===
Gold sentences and facts:
Sentence 0: That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.
  Fact 0.0: That Darn Cat is a film.
  Fact 0.1: That Darn Cat is from 1997.
  Fact 0.2: That Darn Cat is American.
  Fact 0.3: That Darn Cat is a mystery comedy.
  Fact 0.4: That Darn Cat stars Christina Ricci.
  Fact 0.5: That Darn Cat stars Doug E. Doug.
Sentence 1: Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".
  Fact 1.0: Honey, I Blew Up the Kid is a film.
  Fact 1.1: Honey, I Blew Up the Kid is from 1992.
  Fact 1.2: Honey, I Blew Up the Kid is American.
  Fact 1.3: Honey, I Blew Up the Kid is a comedy science fiction film.
  Fact 1.4: Honey, I Blew Up the Kid is a sequel to Honey, I Shrunk the Kids.
  Fact 1.5: Honey, I Shrunk the Kids is a 1989 film.

Retrieved items:
<item 1>
The entries span adult animation, comedy, drama, westerns, science fiction, family films, and television. Ralph Bakshi’s *Fritz the Cat* (1972), the first U.S.
animated feature rated X, and *Fire and Ice* (1983) contrast with his live-action *Cool and the Crazy*. Disney titles include *The Apple Dumpling Gang* films,
*Darby O’Gill*, and *Home on the Range*. Other franchises include *The Naked
</item 1>

<item 2>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 2>

<item 3>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 3>

<item 4>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 4>

<item 5>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 5>

<item 6>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 6>

<item 7>
Honey, We Shrunk Ourselves
Honey, We Shrunk Ourselves is a 1997 live-action direct-to-video sequel to "Honey, I Shrunk the Kids" and "Honey, I Blew Up the Kid".

 It is the third and final installment in the "Honey, I Shrunk the Kids" trilogy.

 The directorial debut of cinematographer Dean Cundey and released through Walt Disney Home Video, it tells the story of inventor Wayne Szalinski as he
accidentally shrinks his wife, brother, sister-in-law, and himself with his electromagnetic shrink ray.
</item 7>

<item 8>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 8>

<item 9>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 9>

<item 10>
Amy O’Neill (born 1971) played Amy Szalinski in *Honey, I Shrunk the Kids* (1989) and its 1992 sequel, earning a Young Artist Award nomination. *Rhoda*, Valerie
Harper’s *Mary Tyler Moore Show* spin-off, ran five seasons (1974–78), followed Rhoda back to New York, and won two Golden Globes and two Emmys. Allison Miller
(born 1985) starred in *K
</item 10>
=== END ITEM id=c9738815aed0985a ===

=== ITEM id=c7468a24c9264377 ===
Gold sentences and facts:
Sentence 0:  In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
  Fact 0.0: In 2003, a person referred to as "he" received an award for his actions during the Battle of Qala-i-Jangi.
  Fact 0.1: The Battle of Qala-i-Jangi took place in late November to early December 2001.
Sentence 1: The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November 25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
  Fact 1.0: The Battle of Qala-i-Jangi is also incorrectly referred to as the "Battle of Mazar-i-Sharif".
  Fact 1.1: The Battle of Qala-i-Jangi was a prisoner-of-war camp uprising.
  Fact 1.2: The Battle of Qala-i-Jangi took place between November 25 and December 1, 2001.
  Fact 1.3: The Battle of Qala-i-Jangi took place in northern Afghanistan.
  Fact 1.4: The Battle of Qala-i-Jangi followed the armed intervention by United States-led coalition forces.
  Fact 1.5: The United States-led coalition forces intervened with armed forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan.
  Fact 1.6: The coalition forces had accused the Taliban's Islamic Emirate of Afghanistan of harboring al-Qaeda operatives.

Retrieved items:
<item 1>
The Great War for Civilisation
The Great War for Civilisation: The Conquest of the Middle East is a book published in 2005 by the award-winning English journalist Robert Fisk.

 The book is a compilation of many of the articles Fisk wrote when he was serving as a correspondent in the Middle East for "The Times" and "The Independent".

 The book revolves around several key themes regarding the history of the modern Middle East: the Arab-Israeli conflict, the Soviet invasion of Afghanistan, the
Persian Gulf War as well as the 2003 Iraq War (United States invasion of Iraq) as well as other regional conflicts such as the Armenian Genocide and the
Algerian Civil War.

 The "Great War for Civilisation" is the second book Fisk has written about the Middle East with the first one, "Pity the Nation", (Nation Books, 2002) being
about the Lebanese Civil War.
</item 1>

<item 2>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 2>

<item 3>
Enemy Combatant (book)
Enemy Combatant is a memoir by British Muslim, Moazzam Begg, co-written by Victoria Brittain, former Associate Foreign Editor for "The Guardian", about Begg's
detention by the government of the United States of America in Bagram Detention Facility and at Camp Echo, Guantanamo Bay and his life prior to that detention.

 It was published in Britain as "Enemy Combatant: A British Muslim's Journey To Guantanamo and Back" (ISBN  ), and in the US as "Enemy Combatant: My
Imprisonment at Guantanamo, Bagram, and Kandahar" (ISBN  ).

 In the US, the foreword was written by David Ignatius of The Washington Post.
</item 3>

<item 4>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 4>

<item 5>
Soldiers of Egypt
Soldiers of Egypt (Ajnad Misr in Arabic) is an active Salafist Islamist militant group that has been operating near Cairo, Egypt.

 The groups was founded by Humam Muhammed in 2013, after he split away from the Ansar Bait al-Maqdis militant group.

 The group claims that its attacks are "retribution" for the August 2013 Rabaa Massacre; notably, the group targets only security forces.

 It has warned civilians of the presence of bombs that it has placed.

 The Cairo Court for Urgent Matters declared the group a terrorist group on 22 May 2014.

 It has been a Proscribed Organisation in the United Kingdom under the Terrorism Act 2000 since November 2014.

 The United States Department of State designated it a terrorist organization on 18 December 2014.
</item 5>

<item 6>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 6>

<item 7>
Combat Terrorist Organization
Combat Terrorist Organization or Autonomous Combat Terrorist Organization (rus.

 Боевая Террористическая Организация) was a Russian radical Neo-Nazi terrorist group based in Saint Petersburg from 9 August 2003 to 2006.

 It was formed by two members of the "Mad Crowd" skinhead organization, Dmitriy Borovikov and Voevodin Alexey.
</item 7>

<item 8>
Iraultza
Iraultza (Basque for Revolution) was a small Basque militant armed group of leftist tendency.

 Seven of the group would die preparing explosive devices.

 On June 27, 1986, a bomb claimed by Iraultza mangled a young worker called José Miguel Moros Peña in Portugalete.

 The victim would die in Cruces Hospital (Barakaldo) on August 13, 1986.

 The group is also known for being described by the United States Government as "probably [having] committed more bombings against American business interests
than any other European terrorist group".

 Soon after, the group, which had been active for about a decade with small attacks of explosives that never aimed to cause any personal injuries, announced its
dissolution.

 When the group was operational, it is estimated that it had approximately 20 members.
</item 8>

<item 9>
Battle of Qala-i-Jangi
The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November
25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's
Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
</item 9>

<item 10>
Sayeed Salahudeen
Syed Mohammed Yusuf Shah, popularly known as Syed Salahudeen, is the head of Hizb-ul-Mujahideen, a terrorist group operating in Kashmir, and head of an alliance
of anti-India militant groups, the United Jihad Council, that works to annex the Indian state of Jammu and Kashmir to Pakistan .

 Salahuddin vowed to block any peaceful resolution to the Kashmir conflict, threatened to train more Kashmiri suicide bombers, and vowed to turn the Kashmir
valley “into a graveyard for Indian forces.”

 He is listed on the NIA Most Wanted list.

 On 26 June 2017, he was named as a Specially Designated Global Terrorist by the US Department Of State.

 After being designated as a Global terrorist, he addressed the Pakistani Media at Muzaffarabad’s Centre Press Club, claiming that "The declaration was a joint
move by the US, Israel, and India to express their animosity towards Pakistan."
</item 10>
=== END ITEM id=c7468a24c9264377 ===
