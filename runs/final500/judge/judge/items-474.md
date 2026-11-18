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

=== ITEM id=b6b1f20201e50323 ===
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
Mike Babcock
Michael "Mike" Babcock Jr. (born April 29, 1963) is a Canadian professional ice hockey head coach, currently serving as head coach of the Toronto Maple Leafs of
the National Hockey League (NHL).

 He previously spent 10 seasons as head coach of the Detroit Red Wings, winning the Stanley Cup with them in 2008 and helping them to the Stanley Cup Playoffs
every year during his tenure.

 He holds the record as the coach with the most wins in Red Wings history, surpassing that of Jack Adams.

 He has also served as head coach of the Mighty Ducks of Anaheim, whom he helped to the 2003 Stanley Cup Finals.
</item 3>

<item 4>
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
</item 4>

<item 5>
1983 Stanley Cup Finals
The 1983 Stanley Cup Finals was contested by the Edmonton Oilers in their first-ever Finals appearance and the defending champion New York Islanders, in their
fourth, and fourth consecutive, Finals appearance.

 The Islanders would win the best-of-seven series four games to none, to win their fourth-straight and fourth-overall Stanley Cup.

 It was also the fourth straight Finals of post-1967 expansion teams, and the first involving a former World Hockey Association (WHA) team.

 This is also the most recent time that a defending Stanley Cup champion has won the cup four years in a row, and also the first (and, to date, only) time a
North American professional sports team has won four consecutive titles in any league competition with more than twenty teams.

 Since 1983, no professional sports team on the continent has managed to win four straight championships and no NHL team has won more than two consecutive
championships (most recently the Pittsburgh Penguins in and ).
</item 5>

<item 6>
1998 Stanley Cup Finals
Detroit coach Scotty Bowman won his eighth Stanley Cup in that capacity (having previously done so with the Montreal Canadiens in , , , , and , the Pittsburgh
Penguins in , and the Wings the previous year), tying him with former Canadiens coach Toe Blake for the record of most Cups won by a coach (which he would break
when he helped the Red Wings win the 2002 Cup).
</item 6>

<item 7>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 7>

<item 8>
Gordon Haidy
Gordon Adam "Gord" Haidy (April 11, 1928 – October 6, 2004) was a professional ice hockey player who played right wing, shooting right.

 He was born in Windsor, Ontario.

 Gordon played only one game, in the semi-finals, for the Stanley Cup champion Detroit Red Wings in 1950.

 Haidy's name was not included on the Stanley Cup, because he was not a regular member of the Red Wings.

 He did qualify to be on the cup, however, so the NHL credits him with winning one Stanley Cup.

 Haidy never played another game in the NHL but played for several more years in the minors and at the senior hockey level for the Ontario Hockey League.

 Haidy is one of only three players who played their only NHL game in the playoffs for a Stanley Cup winning team (See Doug McKay, Chris Hayes).

 Haidy and his teammate Doug McKay accomplished the same feat of the winning both the Calder Cup and Stanley Cup in the same season.
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
</item 10>
=== END ITEM id=b6b1f20201e50323 ===

=== ITEM id=dbc192e65b93377f ===
Gold sentences and facts:
Sentence 0: Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).
  Fact 0.0: Blinding Edge Pictures is an American film production company.
  Fact 0.1: Blinding Edge Pictures was founded in 2000 by M. Night Shyamalan.
  Fact 0.2: Blinding Edge Pictures is known for producing films written and directed by Shyamalan.
  Fact 0.3: Blinding Edge Pictures produced Unbreakable (2000), written and directed by Shyamalan.
  Fact 0.4: Blinding Edge Pictures produced Signs (2002), written and directed by Shyamalan.
  Fact 0.5: Blinding Edge Pictures produced The Village (2004), written and directed by Shyamalan.
  Fact 0.6: Blinding Edge Pictures produced The Happening (2008), written and directed by Shyamalan.
  Fact 0.7: Blinding Edge Pictures produced After Earth (2013), written and directed by Shyamalan.
  Fact 0.8: Blinding Edge Pictures produced The Visit (2015), written and directed by Shyamalan.
  Fact 0.9: Blinding Edge Pictures produced Split (2017), written and directed by Shyamalan.
Sentence 1: Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson, alongside Robin Wright and Spencer Treat Clark.
  Fact 1.0: Unbreakable is a 2000 American superhero thriller film.
  Fact 1.1: Unbreakable was written by M. Night Shyamalan.
  Fact 1.2: Unbreakable was produced by M. Night Shyamalan.
  Fact 1.3: Unbreakable was directed by M. Night Shyamalan.
  Fact 1.4: Unbreakable stars Bruce Willis.
  Fact 1.5: Unbreakable stars Samuel L. Jackson.
  Fact 1.6: Robin Wright is in the cast of Unbreakable.
  Fact 1.7: Spencer Treat Clark is in the cast of Unbreakable.

Retrieved items:
<item 1>
Striking Distance
Striking Distance is a 1993 American action thriller film starring Bruce Willis as Pittsburgh Police homicide detective Thomas Hardy.

 The film co-stars Sarah Jessica Parker, Dennis Farina, and Tom Sizemore.

 It was directed by Rowdy Herrington and written by Herrington and Marty Kaplan.

 The film was shot on location throughout Pittsburgh; its early title was "Three Rivers".
</item 1>

<item 2>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 2>

<item 3>
Blinding Edge Pictures
Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by
Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).

 In 2015, the company released its first television series "Wayward Pines".
</item 3>

<item 4>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 4>

<item 5>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 5>

<item 6>
Unbreakable (film)
Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson,
alongside Robin Wright and Spencer Treat Clark.

 The movie is the first installment in a trilogy.

 In "Unbreakable", a security guard named David Dunn survives a horrific train crash.

 After the incident, with the help of a manipulative disabled comic book shop owner named Elijah Price, he learns that he possesses superhuman powers.

 As Dunn explores and reluctantly confronts his powers while trying to navigate a difficult family life, he begins to fight crime and learns the true nature of
Elijah Price.
</item 6>

<item 7>
Cop Out (2010 film)
Cop Out is a 2010 American buddy cop action-comedy film directed and edited by Kevin Smith, written by Mark and Robb Cullen and starring Bruce Willis, Tracy
Morgan, Kevin Pollak and Seann William Scott.

 The plot revolves around two veteran NYPD partners (Willis and Morgan) on the trail of a stolen, rare, mint-condition baseball card who find themselves up
against a relentless, memorabilia-obsessed bloodthirsty gangster.

 This is the first film that Smith directed that he did not also write.

 Upon its release, the film was met with negative reviews by critics and underperformed at the box office.

 It is nevertheless Kevin Smith's highest grossing film.
</item 7>

<item 8>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 8>

<item 9>
Signs (film)
Signs is a 2002 American science fiction horror film written and directed by M. Night Shyamalan and executive produced by Shyamalan, Frank Marshall, Kathleen
Kennedy and Sam Mercer.

 A joint collective effort to commit to the film's production was made by Blinding Edge Pictures and The Kennedy/Marshall Company.

 It was commercially distributed by Touchstone Pictures theatrically, and by Touchstone Home Entertainment in home media format.

 Its story focuses on a former Episcopal priest named Graham Hess, played by Mel Gibson, who discovers a series of crop circles in his cornfield.

 Hess slowly discovers that the phenomena are a result of extraterrestrial life.

 It also stars Joaquin Phoenix, Rory Culkin, and Abigail Breslin.

 "Signs" explores faith, kinship, and extraterrestrials.
</item 9>

<item 10>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 10>
=== END ITEM id=dbc192e65b93377f ===

=== ITEM id=630c121cbd5de664 ===
Gold sentences and facts:
Sentence 0: The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic Beach.
  Fact 0.0: The Nassau Herald is a weekly newspaper.
  Fact 0.1: The Nassau Herald serves the Five Towns communities of Nassau County.
  Fact 0.2: The Five Towns communities of Nassau County are Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood, and Atlantic Beach.
Sentence 1: Lawrence is a village in Nassau County, New York in the United States.
  Fact 1.0: Lawrence is a village.
  Fact 1.1: Lawrence is in Nassau County, New York, United States.
Sentence 2:  As of the 2010 United States Census, the village population was 6,483.
  Fact 2.0: As of the 2010 United States Census, the population of the village of Lawrence was 6,483.

Retrieved items:
<item 1>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 1>

<item 2>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 2>

<item 3>
Newshold
Newshold SGPS is a Luso-Angolan media group, owned by Pineview Overseas, a Panama-based company, whose shareholders are powerful Angolan figures including
tycoon Alvaro Sobrinho, who represents the Madalena family in this group.

 The company headquarters is in Lisbon.

 It has a share of 15.08% in Cofina, which owns the daily sales leader, the "Correio da Manhã", the largest-circulation Lisbon based tabloid.

 Newshold controls 97% of "Sol", Portugal's third largest weekly newspaper, fully owns daily newspaper "i" and owns minor stakes in two leading magazines,
"Visão" and "Expresso".
</item 3>

<item 4>
Anton Media Group
Anton Media Group, formerly Anton Community Newspapers, and also known by its legal name Long Island Community Newspapers, Inc. is a print media company based
in Mineola, NY on Long Island in Nassau County and produces 17 weekly newspapers in Nassau County as well as "Long Island Weekly" and special sections such as
Healthy Living, Camp & School, and Dining Guides.

 Anton Community Newspapers is one of the largest privately owned newspaper companies in New York State.
</item 4>

<item 5>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 5>

<item 6>
Nassau Herald
The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic
Beach.

 It is part of the Long Island Herald newspaper chain, which includes The Jewish Star and The Oyster Bay Guardian is owned by Richner Communications, and covers
Nassau County, New York.
</item 6>

<item 7>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 7>

<item 8>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 8>

<item 9>
Floral Park, New York
Floral Park is an incorporated village in Nassau County, New York, United States, on Long Island.

 The neighborhood of Floral Park in the New York City borough of Queens, is adjacent to the village.

 The village is at the western border of Nassau County, and is located mainly in the Town of Hempstead, while the section north of Jericho Turnpike is within
the Town of North Hempstead.

 The population as of the US Census of 2010 is 15,863.
</item 9>

<item 10>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 10>
=== END ITEM id=630c121cbd5de664 ===

=== ITEM id=9fdc1a41b74c9da5 ===
Gold sentences and facts:
Sentence 0:  It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke Bader and Linda Riley.
  Fact 0.0: The publication was launched in March 1994.
  Fact 0.1: The publication was launched by Millivres Ltd.
  Fact 0.2: The publication was launched under the editorship of Frances Williams.
  Fact 0.3: The publication is now published by Twin Media Group.
  Fact 0.4: Twin Media Group is owned by Silke Bader.
  Fact 0.5: Twin Media Group is owned by Linda Riley.
Sentence 1: Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.
  Fact 1.0: Elle is a worldwide lifestyle magazine.
  Fact 1.1: Elle is of French origin.
  Fact 1.2: Elle focuses on fashion.
  Fact 1.3: Elle focuses on beauty.
  Fact 1.4: Elle focuses on health.
  Fact 1.5: Elle focuses on entertainment.
Sentence 2:  "Elle" is also the world's best-selling fashion magazine.
  Fact 2.0: Elle is also the world's best-selling fashion magazine.
Sentence 3:  It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.
  Fact 3.0: Elle was founded by Pierre Lazareff.
  Fact 3.1: Elle was founded by Hélène Gordon, the wife of Pierre Lazareff.
  Fact 3.2: Elle was founded in 1945.

Retrieved items:
<item 1>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 1>

<item 2>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 2>

<item 3>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 3>

<item 4>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 4>

<item 5>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 5>

<item 6>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 6>

<item 7>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 7>

<item 8>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 8>

<item 9>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 9>

<item 10>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 10>
=== END ITEM id=9fdc1a41b74c9da5 ===

=== ITEM id=1cf460b72e6d4955 ===
Gold sentences and facts:
Sentence 0: Bank Street () is a short street in Central, Hong Kong.
  Fact 0.0: Bank Street is a short street.
  Fact 0.1: Bank Street is located in Central, Hong Kong.
Sentence 1:  It links Des Voeux Road Central to Queen's Road Central.
  Fact 1.0: Bank Street links Des Voeux Road Central to Queen's Road Central.
Sentence 2:  It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong) respectively.
  Fact 2.0: Bank Street is named after its location between the HSBC Building and the Bank of China Building.
  Fact 2.1: The HSBC Building was the former headquarters of HSBC.
  Fact 2.2: The Bank of China Building was the former headquarters of Bank of China (Hong Kong).
Sentence 3: HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC Holdings.
  Fact 3.0: HSBC Main Building is a headquarters building of The Hongkong and Shanghai Banking Corporation.
  Fact 3.1: The Hongkong and Shanghai Banking Corporation is today a wholly owned subsidiary of London-based HSBC Holdings.

Retrieved items:
<item 1>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 1>

<item 2>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 2>

<item 3>
HSBC lions
Various headquarters and branch buildings of The Hongkong and Shanghai Banking Corporation and the HSBC Group, into which the Hongkong and Shanghai Banking
Corporation has evolved, feature a pair of lion sculptures.

 The HSBC lions have become distinctive landmarks in their own right in Hong Kong and Shanghai respectively, with a further pair to be found in London.
</item 3>

<item 4>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 4>

<item 5>
Bank Street (Hong Kong)
Bank Street () is a short street in Central, Hong Kong.

 It links Des Voeux Road Central to Queen's Road Central.

 It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong)
respectively.
</item 5>

<item 6>
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 6>

<item 7>
Bank of Canton
Bank of Canton () was established in 1912 in Canton but registered as a British company in Hong Kong.

 This made it the first Chinese-owned bank in Hong Kong.

 At the time, all the other banks in Hong Kong were foreign, primarily British, including the locally incorporated but British-run Hongkong and Shanghai Banking
Corporation.

 Security Pacific National Bank (SPNB) bought a majority share in Bank of Canton in 1971.

 In 1988, Security Pacific succeeded in wholly acquiriing Bank of Canton, which became Security Pacific Asia Bank.

 When Bank of America acquired Security Pacific Corporation in 1993 it changed Security Pacific Asia Bank’s name to Bank of America (Asia).

 China Construction Bank acquired Bank of America (Asia) in 2006.
</item 7>

<item 8>
HSBC
HSBC Holdings PLC is a British multinational banking and financial services holding company, tracing its origin to a hong in Hong Kong.

 It is the world's seventh largest bank by total assets and the largest in Europe with total assets of US$2.374 trillion (as of December 2016).

 It was established in its present form in London in 1991 by The Hongkong and Shanghai Banking Corporation Limited to act as a new group holding company.

 The origins of the bank lie mainly in Hong Kong and to a lesser extent in Shanghai, where branches were first opened in 1865.

 The HSBC name is derived from the initials of the Hongkong and Shanghai Banking Corporation.

 The company was first formally incorporated in 1866.

 The company continues to see both the United Kingdom and Hong Kong as its "home markets".
</item 8>

<item 9>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 9>

<item 10>
HSBC Building, the Bund
The HSBC Building (also known as the Municipal Government Building) is a six-floor neo-classical building in the Bund area of Shanghai, China.

 It served as the headquarters of the Shanghai branch of The Hongkong and Shanghai Banking Corporation from 1923 to 1955, and currently houses the Shanghai
Pudong Development Bank.

 The building is situated at number 12, the Bund.

 Designed by the British architecture firm Palmer & Turner Architects and Surveyors, construction of the building lasted from 5 May 1921 to 23 June 1923.
</item 10>
=== END ITEM id=1cf460b72e6d4955 ===

=== ITEM id=0d58fd86f531d1e3 ===
Gold sentences and facts:
Sentence 0: Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale about a wily Irishman and his battle of wits with leprechauns.
  Fact 0.0: Darby O'Gill and the Little People is a feature film.
  Fact 0.1: Darby O'Gill and the Little People was made by Walt Disney Productions.
  Fact 0.2: Darby O'Gill and the Little People was released in 1959.
  Fact 0.3: Darby O'Gill and the Little People stars Albert Sharpe.
  Fact 0.4: Darby O'Gill and the Little People stars Janet Munro.
  Fact 0.5: Darby O'Gill and the Little People stars Sean Connery.
  Fact 0.6: Darby O'Gill and the Little People stars Jimmy O'Dea.
  Fact 0.7: Darby O'Gill and the Little People tells a tale about a wily Irishman.
  Fact 0.8: The Irishman in Darby O'Gill and the Little People has a battle of wits with leprechauns.
Sentence 1:  Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.
  Fact 1.0: The film named in this sentence was produced by Walt Disney Pictures.
  Fact 1.1: The film named in this sentence was produced by Aamir Khan Productions.
  Fact 1.2: The film named in this sentence was produced by UTV Motion Pictures.
  Fact 1.3: Aamir Khan stars as the father in the film named in this sentence.

Retrieved items:
<item 1>
Walt Disney's Treasury of Classic Tales
Walt Disney's Treasury of Classic Tales is an American Sunday comic strip, which ran in newspapers from 13 July 1952 until 15 February 1987.

 Each story adapted a different Disney film, such as Darby O'Gill and the Little People, Peter Pan, or Davy Crockett.

 It was run in relatively few papers with 58 in 1957 and 55 in 1966 and was principally a vehicle for promoting new and re-released Disney films.
</item 1>

<item 2>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 2>

<item 3>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 3>

<item 4>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 4>

<item 5>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 5>

<item 6>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 6>

<item 7>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 7>

<item 8>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 8>

<item 9>
Walt Disney Pictures, founded under that name in 1983, produces live-action films; Walt Disney Studios Motion Pictures distributes Disney studio releases,
succeeding Buena Vista (1953–2007). Disney highlights include *Bambi* (1942), *Pinocchio* (1940), and the 1989–99 Renaissance, spanning ten animated films from
*The Little Mermaid* to *Tarzan*. Buddy Baker scored several Disney films. Other entries cover Mickey and Donald, the
</item 9>

<item 10>
Darby O'Gill and the Little People
Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale
about a wily Irishman and his battle of wits with leprechauns.

 The film was directed by Robert Stevenson and its screenplay written by Lawrence Edward Watkin after the books of Herminie Templeton Kavanagh.
</item 10>
=== END ITEM id=0d58fd86f531d1e3 ===

=== ITEM id=5cb5b8149fb55528 ===
Gold sentences and facts:
Sentence 0: Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.
  Fact 0.0: Pagliacci is an Italian opera.
  Fact 0.1: Pagliacci has a prologue and two acts.
  Fact 0.2: Pagliacci has music and libretto by Ruggero Leoncavallo.
  Fact 0.3: The literal translation of Pagliacci is 'Clowns'.
Sentence 1: Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.
  Fact 1.0: Suor Angelica (Sister Angelica) is an opera.
  Fact 1.1: Suor Angelica is in one act.
  Fact 1.2: Suor Angelica is by Giacomo Puccini.
  Fact 1.3: Suor Angelica has an original Italian libretto by Giovacchino Forzano.

Retrieved items:
<item 1>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 1>

<item 2>
Giovacchino Forzano
Giovacchino Forzano (] ; 19 November 1884 – 28 October 1970) was an Italian playwright, librettist, stage director, and film director.

 A resourceful writer, he authored numerous popular plays and produced opera librettos for most of the major Italian composers of the early twentieth century,
including the librettos for Giacomo Puccini's "Suor Angelica" and "Gianni Schicchi".
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 4>

<item 5>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 5>

<item 6>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 6>

<item 7>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 7>

<item 8>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 8>

<item 9>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 9>

<item 10>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 10>
=== END ITEM id=5cb5b8149fb55528 ===

=== ITEM id=6c5c661e1c98f1a6 ===
Gold sentences and facts:
Sentence 0: The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.
  Fact 0.0: The Battle of Saipan was a battle of the Pacific campaign of World War II.
  Fact 0.1: The Battle of Saipan was fought on the island of Saipan in the Mariana Islands.
  Fact 0.2: The Battle of Saipan was fought from 15 June to 9 July 1944.
Sentence 1:  The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
  Fact 1.0: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division were commanded by Lieutenant General Holland Smith.
  Fact 1.1: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division defeated the 43rd Infantry Division of the Imperial Japanese Army.
  Fact 1.2: The 43rd Infantry Division of the Imperial Japanese Army was commanded by Lieutenant General Yoshitsugu Saito.
Sentence 2:  It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.
  Fact 2.0: The battle referred to here was fought by American and Filipino forces against Japanese troops.
  Fact 2.1: The battle referred to here took place in Manila, the capital city of the Philippines.

Retrieved items:
<item 1>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 1>

<item 2>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 2>

<item 3>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 3>

<item 4>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 4>

<item 5>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 5>

<item 6>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 6>

<item 7>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 7>

<item 8>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 8>

<item 9>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 9>

<item 10>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 10>
=== END ITEM id=6c5c661e1c98f1a6 ===

=== ITEM id=1ef07773cf8db9af ===
Gold sentences and facts:
Sentence 0: War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor at Kean University, and Albie Hecht.
  Fact 0.0: War/Dance is a 2007 American documentary film.
  Fact 0.1: War/Dance was written by Sean Fine and Andrea Nix Fine.
  Fact 0.2: War/Dance was directed by Sean Fine and Andrea Nix Fine.
  Fact 0.3: War/Dance was produced by Shine Global's Susan MacLaury.
  Fact 0.4: Susan MacLaury is a professor at Kean University.
  Fact 0.5: War/Dance was produced by Albie Hecht.
Sentence 1: The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy government and Nazi Germany during World War II.
  Fact 1.0: The Sorrow and the Pity is a documentary film.
  Fact 1.1: The Sorrow and the Pity's French title is Le Chagrin et la Pitié.
  Fact 1.2: The Sorrow and the Pity is in two parts.
  Fact 1.3: The Sorrow and the Pity was released in 1969.
  Fact 1.4: The Sorrow and the Pity was made by Marcel Ophüls.
  Fact 1.5: The Sorrow and the Pity is about the collaboration between the Vichy government and Nazi Germany.
  Fact 1.6: That collaboration occurred during World War II.

Retrieved items:
<item 1>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 1>

<item 2>
Gasland
Gasland is a 2010 American documentary written and directed by Josh Fox.

 Nominated for an Academy Award for Best Documentary in 2011, the film focuses on communities in the United States affected by natural gas drilling and,
specifically, a method of horizontal drilling into shale formations known as hydraulic fracturing.

 The film was a key mobilizer for the anti-fracking movement, and "brought the term 'hydraulic fracturing' into the nation's living rooms" according to "The New
York Times".

 The film was nominated for the Academy Award for Best Documentary Feature.
</item 2>

<item 3>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 3>

<item 4>
Polio v/s Polio victims
Polio v/s Polio Victims is the National Award Winning Documentary film.

 The documentary won "Best Motivational/ Educational Film" at the 56th National Film Awards.

 The documentary follows a group of polio victims who took to the streets in 2008 to spread the message of the Pulse Polio Campaign.

 The film follows them as they go door to door, in various slums in Mumbai, to spread awareness about the upcoming "Pulse Polio Day".
</item 4>

<item 5>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 5>

<item 6>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 6>

<item 7>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 7>

<item 8>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 8>

<item 9>
Christian de la Mazière
Christian de la Mazière (August 22, 1922 in Tunis – February 15, 2006) was a journalist and member of the Charlemagne Division of the Waffen SS.

 He is known for discussing his role in the documentary "The Sorrow and the Pity" and also wrote a book titled "The Captive Dreamer".

 At the start of the war, he served in the French Army and was later retained in the military of Vichy France until 1942.

 After being discharged, he worked for the fascist newspaper "Le Pays Libre", joining the Charlemagne Division just before the Liberation of Paris in 1944.

 He was taken prisoner in Pomerania by Polish forces in the Red Army.
</item 9>

<item 10>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 10>
=== END ITEM id=1ef07773cf8db9af ===

=== ITEM id=d291cf075323860d ===
Gold sentences and facts:
Sentence 0: The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the headquarters of book publisher Random House and a luxury apartment complex.
  Fact 0.0: The Random House Tower is also known as the Park Imperial Apartments.
  Fact 0.1: The Random House Tower is a 52-story mixed-use tower.
  Fact 0.2: The Random House Tower is in New York City, United States.
  Fact 0.3: The Random House Tower is used as the headquarters of book publisher Random House.
  Fact 0.4: The Random House Tower is used as a luxury apartment complex.
Sentence 1:  It currently carries the Vornado Realty Trust corporate headquarters.
  Fact 1.0: The Random House Tower currently carries the Vornado Realty Trust corporate headquarters.

Retrieved items:
<item 1>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 1>

<item 2>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 2>

<item 3>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 3>

<item 4>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 4>

<item 5>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 5>

<item 6>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 6>

<item 7>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 7>

<item 8>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 8>

<item 9>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 9>

<item 10>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 10>
=== END ITEM id=d291cf075323860d ===
