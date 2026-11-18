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

This file has 12 items.

=== ITEM id=62534f8f6572961c ===
Gold sentences and facts:
Sentence 0: During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.
  Fact 0.0: Operation Cold Comfort was a failed SAS raid.
  Fact 0.1: Operation Cold Comfort took place during World War II.
  Fact 0.2: Operation Cold Comfort began with a parachute drop.
  Fact 0.3: The parachute drop of Operation Cold Comfort was north of Verona.
  Fact 0.4: The parachute drop of Operation Cold Comfort was on February 17, 1945.
Sentence 1:  The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.
  Fact 1.0: The SAS was founded in 1941 as a regiment.
  Fact 1.1: The SAS was later reconstituted as a corps.
  Fact 1.2: The SAS was reconstituted as a corps in 1950.

Retrieved items:
<item 1>
Special Air Service
The Special Air Service (SAS) is a special forces unit of the British Army.

 The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.

 The unit undertakes a number of roles including covert reconnaissance, counter-terrorism, direct action and hostage rescue.
</item 1>

<item 2>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 2>

<item 3>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 3>

<item 4>
Operation Cold Comfort
During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.

 It was later renamed Zombie.
</item 4>

<item 5>
Joint Special Operations Command (Jordan)
The Special Operation Forces of the Jordanian Armed Forces serve as Jordan's premiere special forces unit.

 Founded on April 15, 1963 on the orders of the late King Hussein, its primary roles include reconnaissance, counter-terrorism, search and evacuation,
intelligence gathering combat, and the protection of key sites.

 The Special Operation Forces are also charged with carrying out precision strikes against critical enemy targets.

 The 14,000-strong unit are equipped and trained to be able to operate behind enemy lines for long periods without any logistical support, and is considered one
of the finest special forces units in the world.
</item 5>

<item 6>
Special Operations Force (Singapore)
The Special Operations Force (SOF) is the Republic's Army Special Forces composed of highly trained elite soldiers within the Singapore Armed Forces Commando
Formation and an essential component of the joint special forces unit Special Operations Task Force (SOTF).

 According to the Principles of Special Forces, the Republic's special forces thrive on their exceptional qualities and advanced skills, consisting highly
adaptive individuals who are independent and can operate independently, and the Republic's special forces soldiers cannot be mass-produced and must be managed
carefully.
</item 6>

<item 7>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 7>

<item 8>
Operation Barras
Operation Barras was a British Army operation that took place in Sierra Leone on 10 September 2000, during the late stages of that nation's civil war.

 The operation aimed to release five British soldiers of the Royal Irish Regiment who had been held by a militia group known as the "West Side Boys".

 The soldiers were part of a patrol that was returning from a visit to Jordanian peacekeepers attached to the United Nations Mission in Sierra Leone (UNAMSIL)
at Masiaka on 25 August 2000 when they turned off the main road and down a track towards the village of Magbeni.

 There the patrol was overwhelmed by a large number of heavily armed rebels, taken prisoner, and transported to Gberi Bana on the opposite side of Rokel Creek.
</item 8>

<item 9>
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
</item 9>

<item 10>
Jan Breytenbach
Jan Dirk Breytenbach {'1': ", '2': ", '3': ", '4': "} (born 4 July 1933) was appointed by General Fritz Loots, the founder of the South African Special Forces
Brigade, as the first commander of 1 Reconnaissance Commando, the first unit founded within the South African Special Forces.

 He was also appointed as the first commander of the 32 Battalion, known colloquially as "Buffalo Battalion", as well as 44 Parachute Brigade.
</item 10>
=== END ITEM id=62534f8f6572961c ===

=== ITEM id=d8847b31e1716143 ===
Gold sentences and facts:
Sentence 0: Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.
  Fact 0.0: Victor Ronald Salva was born March 29, 1958.
  Fact 0.1: Victor Ronald Salva is an American film director.
  Fact 0.2: Victor Ronald Salva is an American screenwriter.
Sentence 1: Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.
  Fact 1.0: Emilio "El Indio" Fernández was born Emilio Fernández Romo.
  Fact 1.1: Emilio "El Indio" Fernández was born March 26, 1904.
  Fact 1.2: Emilio "El Indio" Fernández died August 6, 1986.
  Fact 1.3: Emilio "El Indio" Fernández was a Mexican film director.
  Fact 1.4: Emilio "El Indio" Fernández was a Mexican actor.
  Fact 1.5: Emilio "El Indio" Fernández was a Mexican screenwriter.

Retrieved items:
<item 1>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 1>

<item 2>
Emilio Fernández
Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.

 He was one of the most prolific film directors of the Golden Age of Mexican cinema in the 1940s and 1950s.

 He is best known for his work as director of the film "Maria Candelaria" (1944), which won the Palme d'Or award at the 1946 Cannes Film Festival.

 As an actor, he worked in numerous film productions in Mexico and in Hollywood.
</item 2>

<item 3>
Columba Domínguez
Columba Domínguez Adalid (March 4, 1929 – August 13, 2014) was a Mexican actress, singer, and painter.

 Considered a crucial figure in the Golden Age of Mexican cinema, she was one of the muses of the film director Emilio Fernández, with whom she was romantically
linked for several years.

 She is remembered particularly for her performance in the film "Pueblerina" (1949), which is considered one of the jewels of the Mexican Cinema.
</item 3>

<item 4>
Río Escondido (1948 film)
Río Escondido is a 1948 Mexican drama film directed by Emilio Fernández and starring María Félix.
</item 4>

<item 5>
Jaime Fernández (actor)
Jaime Fernández Reyes (December 6, 1937 – April 15, 2005) was a Mexican actor.

 Over his career, he won 3 Silver Ariel awards - the Mexican equivalent of the Oscar - including one for what is arguably his best-known role, playing Friday in
Luis Buñuel's "Robinson Crusoe".

 He appeared in over 200 films and served as the General Secretary of the Mexican actors' union for 11 years.

 His older brother was actor/director Emilio Fernández.
</item 5>

<item 6>
Enamorada (film)
Enamorada ("Enamoured") is a 1946 Mexican drama film directed by Emilio Fernández and starring María Félix and Pedro Armendáriz.

 The film was remade in 1950 as "The Torch" with Armendáriz repeating his role alongside Paulette Goddard who was credited as an associate producer on the film.
</item 6>

<item 7>
Fernando Fernández (actor)
Fernando Fernández "El Crooner de México" (1916–1999) was a Mexican actor and singer.

 He was born on November 9, 1916, in Monterrey, Mexico, and died in Mexico City, Mexico, on November 24, 1999.

 Fernando was the son of Eloisa Reyes.

 He was brother of the famous film director Emilio Fernández and actor Jaime Fernández.

 He was married to singer Lupita Palomera, who died in 2008.

 Fernando Fernández was known as "the Crooner of Mexico."
</item 7>

<item 8>
Comedy in the Golden Age of Mexican cinema
Film became popular and well known in Mexico during “El Porfiriato” which represents the reign of Porfirio Díaz as dictator in Mexico.

 On August 6, 1896 the first films began to be released and were called/known as “Vistas”.

 In other words, views.

 At the time, the Catholic Church and the press were strongly opposed to the “vistas” because of the violence and women that was portrayed.

 They protested against them.

 Another important aspect was the projection of the Mexican Revolution to the public through film.

 It was a way for many to understand what was happening politically.

 Many agree that rather than the Mexican Revolution being a repression for the film industry, it was more of a motivator in the way it portrayed every battle
and struggle.

 With the positive energy that the Mexico’s film industries was receiving, it began to evolve into “El Cine Sonoro” which translates to the cinema of sound.

 This was the era in which sound was being added into the silent films.

 Sadly, it was not as successful as many directors and film maker though it would be.
</item 8>

<item 9>
Las Abandonadas
Las Abandonadas (The Abandoned) is a Mexican film of 1945, directed by Emilio Fernández, starring Dolores del Río and Pedro Armendáriz.
</item 9>

<item 10>
Comedy in the Golden Age of Mexican cinema
Sadly, it was not as successful as many directors and film maker though it would be.

 Many failed to correctly synchronize the audio with the picture.

 Although to some extent it may have been unsuccessful, it is agreeable that it helped revolutionize film in Latin America.

 Mexico’s film industry takes its height post “El Cine Sonoro” to what is known as “The Golden Age”.

 Many film members consider this one of the most successful eras in the history of Mexican cinema.

 Many recognize the films of this era when they see black and white.

 It is also important to note that the films also portrayed social issues.

 Some of the most influential directors and actors are Emilio Fernández, Raul De Anda, Roberto B. Cervenna, Cantinflas, Vitola, Tin Tan, etc.

 The focus of this article will be on some of top grossing comedian actors of the time Cantinflas, Tin Tan, Clavillazo and others.
</item 10>
=== END ITEM id=d8847b31e1716143 ===

=== ITEM id=c44e61110a947fbb ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Nicolas Cage starred in a variety of films.
  Fact 0.1: Nicolas Cage starred in Valley Girl (1983).
  Fact 0.2: Nicolas Cage starred in Racing with the Moon (1984).
  Fact 0.3: Nicolas Cage starred in Birdy (1984).
  Fact 0.4: Nicolas Cage starred in Peggy Sue Got Married (1986).
  Fact 0.5: Nicolas Cage starred in Raising Arizona (1987).
  Fact 0.6: Nicolas Cage starred in Moonstruck (1987).
  Fact 0.7: Nicolas Cage starred in Vampire's Kiss (1989).
  Fact 0.8: Nicolas Cage starred in Wild at Heart (1990).
  Fact 0.9: Nicolas Cage starred in Fire Birds (1990).
  Fact 0.10: Nicolas Cage starred in Honeymoon in Vegas (1992).
  Fact 0.11: Nicolas Cage starred in Red Rock West (1993).
Sentence 1:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 1.0: The film is about a widowed 37-year-old Italian-American woman.
  Fact 1.1: Cher plays the widowed 37-year-old Italian-American woman.
  Fact 1.2: The widowed woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 1.3: The widowed woman's fiancé is played by Danny Aiello.
  Fact 1.4: The fiancé's younger brother is estranged.
  Fact 1.5: The fiancé's younger brother is hot-tempered.
  Fact 1.6: The fiancé's younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
He Knew Women
He Knew Women is a 1930 American comedy film, directed by Hugh Herbert, from a screenplay by him and William B. Jutte, which was adapted from S. N. Behrman's
1927 play "The Second Man".

 It starred Lowell Sherman and Alice Joyce, in her second to last film role.

 The film just broke even.
</item 1>

<item 2>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 2>

<item 3>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 3>

<item 4>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 4>

<item 5>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 5>

<item 6>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 6>

<item 7>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 7>

<item 8>
A Star Is Born (1976 film)
A Star Is Born is a 1976 American musical drama film telling the story of a young woman, played by Barbra Streisand, who enters show business, and meets and
falls in love with an established male star, played by Kris Kristofferson, only to find her career ascending while his goes into decline.

 It is a remake of two earlier versions – the 1937 version was a drama starring Janet Gaynor and Fredric March, and the 1954 version was a musical starring Judy
Garland and James Mason.

 It will be remade for a third time in 2018 starring Stefani Germanotta and Bradley Cooper.
</item 8>

<item 9>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 9>

<item 10>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 10>
=== END ITEM id=c44e61110a947fbb ===

=== ITEM id=e402486221fb7ea3 ===
Gold sentences and facts:
Sentence 0:  is a board game created by Glenn Drover and published by Eagle Games in 2003.
  Fact 0.0: The board game in this sentence was created by Glenn Drover.
  Fact 0.1: The board game in this sentence was published by Eagle Games in 2003.
Sentence 1: Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.
  Fact 1.0: Can't Stop is a board game designed by Sid Sackson.
  Fact 1.1: Can't Stop was originally published by Parker Brothers in 1980.
  Fact 1.2: Can't Stop was long out of print in the United States.

Retrieved items:
<item 1>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 1>

<item 2>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 2>

<item 3>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 3>

<item 4>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 4>

<item 5>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 5>

<item 6>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 6>

<item 7>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 7>

<item 8>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 8>

<item 9>
Francis Tresham (game designer)
Francis Tresham is a United Kingdom-based board game designer who has been producing board games since the early 1970s.

 Tresham founded and ran games company Hartland Trefoil (founded 1971), a company well known for its "Civilization" board game, until its sale to MicroProse in
1997.

 His "1829" game was the first of the "18xx" board game series and some of his board games have inspired Sid Meier computer games such as "Railroad Tycoon".
</item 9>

<item 10>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 10>
=== END ITEM id=e402486221fb7ea3 ===

=== ITEM id=e050a44c0a93c77c ===
Gold sentences and facts:
Sentence 0:  It has an area of 1208 km2 and had a population of 600,000 in 2003.
  Fact 0.0: Ezhou has an area of 1208 km2.
  Fact 0.1: Ezhou had a population of 600,000 in 2003.
Sentence 1: Ezhou () is a prefecture-level city in eastern Hubei Province, China.
  Fact 1.0: Ezhou is a prefecture-level city.
  Fact 1.1: Ezhou is in eastern Hubei Province.
  Fact 1.2: Ezhou is in China.
Sentence 2:  As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.
  Fact 2.0: As of the 2010 census, Ezhou had a population of 1,048,668.
  Fact 2.1: As of the 2010 census, 668,727 of Ezhou's residents lived in the core Echeng District.
Sentence 3:  The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
  Fact 3.0: The Ezhou-Huanggang built-up (or metro) area was home to 1,035,496 inhabitants.
  Fact 3.1: The inhabitants of the Ezhou-Huanggang built-up area came from the Echeng and Huangzhou districts of Huanggang.

Retrieved items:
<item 1>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 1>

<item 2>
Dafeng District
Dafeng () is a coastal district under the administration of Yancheng, Jiangsu province, China.

 Located on the Jiangsu North Plain with a coastline of 112 km , Dafeng was historically one of the largest salt-making areas in China and now is famed for its
well preserved eco-system and numerous national conservation parks.

 The district has the largest national nature reserve for a rare deer species, Père David's Deer or Milu (麋鹿 ) in Chinese.

 It borders the prefecture-level city of Taizhou to the southwest.
</item 2>

<item 3>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 3>

<item 4>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 4>

<item 5>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 5>

<item 6>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 6>

<item 7>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 7>

<item 8>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 8>

<item 9>
Jiangyin dialect
Jiangyin dialect (江阴话) is a Northern Wu Chinese dialect spoken in the city of Jiangyin in Jiangsu province.

 Jiangyin dialect is a member of the Wu Chinese Taihu Wu family of dialects, which means the inhabitants speak a dialect similar to that of nearby Wuxi,
Changzhou, Suzhou, and Shanghai.

 Jiangyin dialect itself is of the Piling variety, related to the Changzhou dialect.

 Jiangyin dialect has the highest degree of mutual intelligibility with the dialects of the closest neighboring cities of Changzhou and Wuxi but also has a
fairly large degree of mutual intelligibility with the dialects of nearby Suzhou and Shanghai.

 As one travels south towards Wuxi away from the urban center of Jiangyin, Jiangyin dialect gradually becomes more and more closer sounding to the Wuxi dialect.
</item 9>

<item 10>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 10>
=== END ITEM id=e050a44c0a93c77c ===

=== ITEM id=9c63412997f02ed2 ===
Gold sentences and facts:
Sentence 0: Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in 2011.
  Fact 0.0: Pramod Bhasin stepped down as President of Genpact.
  Fact 0.1: Pramod Bhasin stepped down as CEO of Genpact.
  Fact 0.2: Genpact is India's largest business process outsourcing (BPO) company.
  Fact 0.3: Pramod Bhasin became non-executive Vice Chairman in 2011.
Sentence 1: Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.
  Fact 1.0: Genpact is a global business process management and services corporation.
  Fact 1.1: Genpact is an IT corporation.
  Fact 1.2: Genpact has key offices in New York City, United States.

Retrieved items:
<item 1>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 1>

<item 2>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 2>

<item 3>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 3>

<item 4>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 4>

<item 5>
Mike Long (American businessman)
Mike Long is an American business man, former CEO of several public companies, and currently a founding partner of Sulgrave Partners LLC.

 He served as the President and Chief Executive Officer of Continuum, an Austin, Texas IT consulting company, from 1991 to 1997, having started with Continuum
as a Director in 1983.

 In 1997, Long was named CEO of Healtheon Corporation (now WebMD), succeeding former CEO David Schnell.

 Long oversaw Healtheon's initial public offering, traveling between Europe and the United States to woo investors.

 Long was able to secure the required investment funds, and saw Healtheon's stock price rise from $8 to a high of $120.

 In 2002, Long was recruited to fix the financial struggles of Move, Inc., a company plagued by more than $4 billion in lawsuits and hemorrhaging tens of
millions of dollars a quarter.

 As Chief Executive Officer, Long was able to revive Homestore, Inc., by changing the business model, rebranding the company as Move, Inc. and returning it to
profitability.
</item 5>

<item 6>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 6>

<item 7>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 7>

<item 8>
Pramod Bhasin
Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in
2011.
</item 8>

<item 9>
Mike Long (American businessman)
and returning it to profitability.

 Touching on his experience of bringing about the initial public offerings of web-based businesses, Long would say that investors needed to be presented "with
an entirely new face every few months," and that "the only way to run one of these Silicon Valley companies was to forget everything you'd learned outside of
Silicon Valley."
</item 9>

<item 10>
Summit Behavioral Healthcare
Summit Behavioral Healthcare (Summit BHC) is a Brentwood, Tennessee-based behavioral health services company that owns and operates a large network of addiction
treatment centers throughout the United States.

 Originating in 2012, the company was founded by Trey Carter, former CEO of Acadia Healthcare.

 In March 2015, Chicago, IL based investment firm Flexpoint Ford provided Round 1 funding to launch Summit BHC's growth plan.
</item 10>
=== END ITEM id=9c63412997f02ed2 ===

=== ITEM id=0f7ccff91b4781a7 ===
Gold sentences and facts:
Sentence 0: Ecopack Ltd is a PET bottles blowing company in Pakistan.
  Fact 0.0: Ecopack Ltd is a PET bottles blowing company.
  Fact 0.1: Ecopack Ltd is located in Pakistan.
Sentence 1: Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and in combination with glass fiber for engineering resins.
  Fact 1.0: Polyethylene terephthalate is also written as poly(ethylene terephthalate).
  Fact 1.1: Polyethylene terephthalate is commonly abbreviated as PET.
  Fact 1.2: Polyethylene terephthalate is commonly abbreviated as PETE.
  Fact 1.3: Polyethylene terephthalate is sometimes abbreviated as PETP, which is obsolete.
  Fact 1.4: Polyethylene terephthalate is sometimes abbreviated as PET-P, which is obsolete.
  Fact 1.5: Polyethylene terephthalate is the most common thermoplastic polymer resin of the polyester family.
  Fact 1.6: Polyethylene terephthalate is used in fibers for clothing.
  Fact 1.7: Polyethylene terephthalate is used in containers for liquids and foods.
  Fact 1.8: Polyethylene terephthalate is used in thermoforming for manufacturing.
  Fact 1.9: Polyethylene terephthalate is used in combination with glass fiber for engineering resins.

Retrieved items:
<item 1>
Polytrimethylene terephthalate
Polytrimethylene terephthalate (PTT), is a polyester synthesized and patented in 1941.

 It is produced by a method called condensation polymerization or transesterification.

 The two monomer units used in producing this polymer are: 1,3-propanediol and terephthalic acid or dimethyl terephthalate.

 Similar to polyethylene terephthalate, the PTT is used to make carpet fibers.
</item 1>

<item 2>
Polyethylene naphthalate
Polyethylene naphthalate (poly(ethylene 2,6-naphthalate or PEN) is a polyester with good barrier properties (even better than Polyethylene terephthalate).

 Because it provides a very good oxygen barrier, it is particularly well-suited for bottling beverages that are susceptible to oxidation, such as beer.

 It is also used in making high performance sailcloth.

 It also has been found to show supreme scintillation properties and is expected to replace classic plastic scintillators.
</item 2>

<item 3>
Ecopack
Ecopack Ltd is a PET bottles blowing company in Pakistan.
</item 3>

<item 4>
Polycyclohexylenedimethylene terephthalate
Polycyclohexylenedimethylene terephthalate (PCT) is a thermoplastic polyester formed from the polycondensation of terephthalic acid and cyclohexanedimethanol.

Its chemical structure is similar to that of polyethylene terephthalate (PET), with which it shares properties like dimensional stability and chemical
resistance.

 PCT is also particularly resistant to high temperatures and hydrolysis.

 The melting point is 545 °F (285 °C).

 Common brand names are "Thermx" (Ticona), "Eastar" (Eastman) and "SkyPURA" (SK Chemicals).
</item 4>

<item 5>
Krones
Krones AG is a German packaging and bottling machine manufacturer.

 It is the world's leading manufacturer of lines for filling beverages in plastic and glass bottles or beverage cans.

 The company manufactures stretch blow-moulding machines for producing polyethylene terephthalate (PET) bottles, plus fillers, labellers, bottle washers,
pasteurisers, inspectors, packers and palletisers.

 This product portfolio is complemented by material flow systems and process technology for producing beverages, plus syrup kitchens, for clients like
breweries, dairies and soft-drink companies.
</item 5>

<item 6>
BoPET
BoPET (Biaxially-oriented polyethylene terephthalate) is a polyester film made from stretched polyethylene terephthalate (PET) and is used for its high tensile
strength, chemical and dimensional stability, transparency, reflectivity, gas and aroma barrier properties, and electrical insulation.
</item 6>

<item 7>
Xenoy
Xenoy is a blend of plastics with many industrially-useful properties.

 It is typically polyester (polybutylene terephthalate, PBT, or polyethylene terephthalate, PET) and polycarbonate (PC): it is often labeled PBT+PC or PET+PC.
</item 7>

<item 8>
Polyethylene 2,5-furandicarboxylate
Polyethylene 2,5-furandicarboxylate, also named poly(ethylene 2,5-furandicarboxylate), polyethylene furanoate and poly(ethylene furanoate) and generally
abbreviated as PEF, is a polymer that can be produced by polycondensation of 2,5-furandicarboxylic acid (FDCA) and monoethylene glycol (MEG).

 As an aromatic polyester from ethylene glycol it is a chemical analogue of polyethylene terephthalate (PET) and polyethylene naphthalate (PEN).

 PEF has been described in (patent) literature since 1951, but has gained renewed attention since the US department of energy proclaimed its building block,
FDCA, as a potential bio-based replacement for purified terephthalic acid (PTA) in 2004.
</item 8>

<item 9>
Two-liter bottle
The two-liter bottle is a common container for soft drinks.

 These bottles are produced from polyethylene terephthalate, also known as PET plastic, using the blow molding process.

 Bottle labels consist of a printed, tight-fitted plastic sleeve.

 A resealable screw-top allows the contents to be used at various times while retaining carbonation.
</item 9>

<item 10>
Polyethylene terephthalate
Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common
thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and
in combination with glass fiber for engineering resins.
</item 10>
=== END ITEM id=0f7ccff91b4781a7 ===

=== ITEM id=5de2ec82c4b0eadb ===
Gold sentences and facts:
Sentence 0: John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.
  Fact 0.0: John Alexander was born on 5 October 1955.
  Fact 0.1: John Alexander is English.
  Fact 0.2: John Alexander is a former footballer.
  Fact 0.3: John Alexander holds the post of club secretary at Manchester United.
Sentence 1:  Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in 1976.
  Fact 1.0: John Alexander was born in Liverpool.
  Fact 1.1: John Alexander began his football career with a club called Ulysses.
  Fact 1.2: John Alexander was later picked up by Millwall.
  Fact 1.3: John Alexander made his league debut for Millwall in 1976.
Sentence 2:  He is the uncle of Liverpool player Trent Alexander-Arnold.
  Fact 2.0: John Alexander is the uncle of Liverpool player Trent Alexander-Arnold.
Sentence 3: Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.
  Fact 3.0: Trent John Alexander-Arnold was born on 7 October 1998.
  Fact 3.1: Trent John Alexander-Arnold is English.
  Fact 3.2: Trent John Alexander-Arnold is a professional footballer.
  Fact 3.3: Trent John Alexander-Arnold plays as a right-back.
  Fact 3.4: Trent John Alexander-Arnold plays for Premier League club Liverpool.

Retrieved items:
<item 1>
Trent Alexander-Arnold
Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.

 He is an academy graduate of Liverpool and made his senior debut for the club in 2016, aged 18.

 The following year, he won the club's Young Player of the Season award.

 He has also represented England at all youth levels from under-16 through to under-19.

 Due to the length of his name, his surname is shortened to Alexander for Premier League fixtures.
</item 1>

<item 2>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 2>

<item 3>
Wilton Lines
Wilton Lines (fl.

 1889–1890) was an English professional footballer born in Birmingham.

 Lines played seven games for Small Heath in the inaugural 1889–90 season of the Football Alliance, covering a variety of forward positions, but was unable to
displace the established forwards.
</item 3>

<item 4>
John Logan (footballer, born 1912)
John William Logan (16 August 1912 – October 1980) was an English professional footballer born in Horden, near Peterlee, County Durham who played in the
Football League for Darlington, Barnsley and Sheffield Wednesday.

 He played as a wing half.
</item 4>

<item 5>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 5>

<item 6>
Helen Atkinson-Wood
Helen Atkinson-Wood (born 14 March 1955) is an English actress and comedian born in Cheadle Hulme, Cheshire.
</item 6>

<item 7>
John Alexander (footballer, born 1955)
John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.

 Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in
1976.

 After scoring twice in 15 appearances over the course of two years with Millwall, he moved to Reading; he enjoyed greater success with Reading, scoring nine
goals in 25 games.

 In 1981, he joined Northampton Town, but spent just one season with them before retiring from football at the age of 26 in 1982.

 Upon retiring from football, Alexander got a job with the BBC, but he later returned to football as club secretary at Watford.

 In 2000, he took up the same post at Tottenham Hotspur, spending 10 years there before applying for the same job at Manchester United, where he would replace
the retiring Ken Ramsden.

 He took over at Manchester United on 1 July 2010.

 He is the uncle of Liverpool player Trent Alexander-Arnold.
</item 7>

<item 8>
Nick Powell
Nicholas Edward "Nick" Powell (born 23 March 1994) is an English professional footballer who plays for Wigan Athletic as a midfielder.

 He began his career at Crewe Alexandra, initially as a forward, making his debut at the age of 16.

 After impressing for Crewe Alexandra during the 2011–12 season, including scoring in their 2–0 victory in the League Two play-off Final, he moved to Manchester
United in July 2012.

 Unable to break into the Manchester United first team, however, he went on loan to Wigan Athletic, Leicester City and Hull City before being released in June
2016.

 He joined Wigan Athletic in July 2016.

 He has represented England at under-16, under-17, under-18, under-19 and under-21 levels.
</item 8>

<item 9>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 9>

<item 10>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 10>
=== END ITEM id=5de2ec82c4b0eadb ===

=== ITEM id=808f0d5d5e3dd968 ===
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
Mario Lemieux
Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.

 He played parts of 17 seasons in the National Hockey League's (NHL) with the Pittsburgh Penguins, between 1984 and 2006.

 Dubbed "The Magnificent One" or "Le Magnifique" (as well as "Super Mario"), he is widely acknowledged to have been one of the best players of all time.

 He is the only player to score one goal in each of the five possible situations in a single NHL game, a feat he accomplished in 1988.

 A gifted playmaker and fast skater despite his large size, Lemieux often beat defencemen with fakes and dekes.
</item 1>

<item 2>
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
</item 2>

<item 3>
1998 Stanley Cup Finals
Detroit coach Scotty Bowman won his eighth Stanley Cup in that capacity (having previously done so with the Montreal Canadiens in , , , , and , the Pittsburgh
Penguins in , and the Wings the previous year), tying him with former Canadiens coach Toe Blake for the record of most Cups won by a coach (which he would break
when he helped the Red Wings win the 2002 Cup).
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
1984 Stanley Cup Finals
This would be the second of eight consecutive Finals contested by a team from Alberta (the Oilers appeared in six, the Calgary Flames in two), and the first of
five consecutive Finals to end with the Cup presentation on Alberta ice (the Oilers won four times, the Montreal Canadiens one).
</item 5>

<item 6>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 6>

<item 7>
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
</item 7>

<item 8>
1917–18 Toronto Hockey Club season
The 1917–18 Toronto Hockey Club season was the first season of the new "Toronto" franchise in the newly organized National Hockey League (NHL).

 The team was intended as a 'temporary' franchise, operating without an official club nickname (the press would dub them the "Blue Shirts" or "Torontos", and in
1948 the NHL would engrave "Toronto Arenas" on the Stanley Cup as the 1917–18 winner) and without a formal organization separate from the Toronto Arena Company
that managed the Arena Gardens.

 Despite this, the team came together to win the first NHL Championship, competing against existing teams that had transferred directly from the National Hockey
Association (NHA).

 Toronto would go on to win the Stanley Cup by defeating the Pacific Coast Hockey Association champion Vancouver Millionaires – the first Stanley Cup for an NHL
team and the second Cup for a Toronto team after the Toronto Blueshirts' victory in the 1913–14 season of the NHA.
</item 8>

<item 9>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 9>

<item 10>
1983 Stanley Cup Finals
The 1983 Stanley Cup Finals was contested by the Edmonton Oilers in their first-ever Finals appearance and the defending champion New York Islanders, in their
fourth, and fourth consecutive, Finals appearance.

 The Islanders would win the best-of-seven series four games to none, to win their fourth-straight and fourth-overall Stanley Cup.

 It was also the fourth straight Finals of post-1967 expansion teams, and the first involving a former World Hockey Association (WHA) team.

 This is also the most recent time that a defending Stanley Cup champion has won the cup four years in a row, and also the first (and, to date, only) time a
North American professional sports team has won four consecutive titles in any league competition with more than twenty teams.

 Since 1983, no professional sports team on the continent has managed to win four straight championships and no NHL team has won more than two consecutive
championships (most recently the Pittsburgh Penguins in and ).
</item 10>
=== END ITEM id=808f0d5d5e3dd968 ===

=== ITEM id=855e73f7246485e8 ===
Gold sentences and facts:
Sentence 0: Eryngium is a genus of flowering plants in the family Apiaceae.
  Fact 0.0: Eryngium is a genus of flowering plants.
  Fact 0.1: Eryngium is in the family Apiaceae.
Sentence 1: Arnebia is a genus of flowering plants in the family Boraginaceae.
  Fact 1.0: Arnebia is a genus of flowering plants.
  Fact 1.1: Arnebia is in the family Boraginaceae.

Retrieved items:
<item 1>
Eucryphia: seven mostly evergreen South American/Australian trees; classified in Cunoniaceae. Bothriochloa ischaemum: widespread grass; var. songarica invasive
in Texas. Eriogonum buckwheats: genus diverse across North America; E. heracleoides inhabits western U.S.; E. diclinum forms low mats on serpentine soils; E.
crosbyae occurs in Oregon/Nevada.
</item 1>

<item 2>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 2>

<item 3>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 3>

<item 4>
Huernia is an African stem-succulent genus whose carrion-scented flowers attract flies. The Asian emerald ash borer devastates ash trees abroad. Oenanthe
javanica is an Asian water vegetable. California botanist Samuel Parish is commemorated by many plant names. Ardisia, Myrsine and Rapanea are Primulaceae
genera; Broughtonia is Caribbean orchids; Stephanocereus Brazilian cacti; Bognera a rare
</item 4>

<item 5>
Cyanobiont
Several cyanobionts involved with fungi and marine organisms also belong to the genera "Richelia, Calothrix, Synechocystis, Aphanocapsa" and "Anabaena", as well
as the species "Oscillatoria spongeliae".

 Although there are many documented symbioses between cyanobacteria and marine organisms, little is known about the nature of many of these symbioses.

 The possibility of discovering more novel symbiotic relationships is apparent from preliminary microscopic observations.
</item 5>

<item 6>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 6>

<item 7>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 7>

<item 8>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 8>

<item 9>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 9>

<item 10>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 10>
=== END ITEM id=855e73f7246485e8 ===

=== ITEM id=b52b350d7e9be491 ===
Gold sentences and facts:
Sentence 0: James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more than 70 productions to his credit.
  Fact 0.0: James David Sharman was born on 12 March 1945.
  Fact 0.1: James David Sharman is the son of boxing tent entrepreneur Jimmy Sharman.
  Fact 0.2: James David Sharman is an Australian director.
  Fact 0.3: James David Sharman is an Australian writer for film and stage.
  Fact 0.4: James David Sharman has more than 70 productions to his credit.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born on August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is an American director.
  Fact 1.2: Ava Marie DuVernay is an American screenwriter.
  Fact 1.3: Ava Marie DuVernay is an American film marketer.
  Fact 1.4: Ava Marie DuVernay is an American film distributor.

Retrieved items:
<item 1>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 1>

<item 2>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 2>

<item 3>
Albert Sharpe
Albert Sharpe (15 April 1885 – 13 February 1970) was an Irish stage and film actor.

 His most famous roles were those of Darby O'Gill in Disney's "Darby O'Gill and the Little People" and as Finian McLonergan in the Original Broadway production
of the musical "Finian's Rainbow".

 (the film version, made in 1968, had Fred Astaire in the role.)

 On screen he played Fiona's father Andrew in the MGM musical Brigadoon.

 He was also a member of the Abbey Players.

 His last ten years were spent in retirement.

 He died in 1970 in Belfast at the age of 84.
</item 3>

<item 4>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 4>

<item 5>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 5>

<item 6>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 6>

<item 7>
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 7>

<item 8>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 8>

<item 9>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 9>

<item 10>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 10>
=== END ITEM id=b52b350d7e9be491 ===

=== ITEM id=37d89418adeaf353 ===
Gold sentences and facts:
Sentence 0: Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.
  Fact 0.0: Emmitt H. Ford was born on December 13, 1943.
  Fact 0.1: Emmitt H. Ford died on November 10, 2014.
  Fact 0.2: Emmitt H. Ford was an American politician.
  Fact 0.3: Emmitt H. Ford was from Tennessee.
Sentence 1:  He was also an uncle of former United States Congressman Harold Ford Jr.
  Fact 1.0: Emmitt H. Ford was an uncle of Harold Ford Jr.
  Fact 1.1: Harold Ford Jr. is a former United States Congressman.
Sentence 2:  He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.
  Fact 2.0: Emmitt H. Ford is the son of Harold Ford Sr.
  Fact 2.1: Harold Ford Sr. is a former Congressman.
  Fact 2.2: Harold Ford Sr. held the same seat for 22 years.

Retrieved items:
<item 1>
Emmitt Ford
Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.

 He represented the 86th district encompassing Shelby County from 1975–1981.

 He was also an uncle of former United States Congressman Harold Ford Jr.
</item 1>

<item 2>
United States Senate election in Tennessee, 2006
The 2006 United States Senate election in Tennessee was held on November 7, 2006.

 The election winner, Bob Corker, served his term between January 3, 2007 and January 3, 2013.

 Corker replaced Republican Bill Frist in the Senate, who retired upon the end of his second term in 2007.

 Corker was the Republican nominee, and the Democratic nominee was Harold Ford, Jr..

 The race between Ford and Corker was one of the most competitive Senate races of 2006, with Corker winning the race by less than three percent of the vote.

 Corker was the only non-incumbent Republican to win a U.S. Senate seat in 2006.

 Since 1994, the Republican Party has held both of Tennessee's Senate seats.
</item 2>

<item 3>
Doug Ford Jr.
Douglas Bruce Ford Jr. (born November 20, 1964) is a Canadian businessman and politician in Toronto, Ontario.

 Ford was Toronto City Councillor for Ward 2 Etobicoke North in Toronto from 2010 to 2014 at the same time that his brother, Rob Ford (1969–2016), was mayor of
Toronto.

 Their father, Doug Ford Sr., was a Member of Provincial Parliament from 1995 to 1999 and founded Deco Labels and Tags, a printing business operating in Canada
and the United States.

 The business is now owned by Doug Ford Jr., who has served as company president since 2002, and his brother, Randy.
</item 3>

<item 4>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 4>

<item 5>
John Ford (Tennessee politician)
John N. Ford (born May 3, 1942 in Memphis, Tennessee), is a former Democratic member of the Tennessee State Senate and a member of Tennessee's most prominent
African-American political family.

 He is the older brother of former U.S. Representative Harold Ford, Sr. and the uncle of former Tennessee U.S. Representative and 2006 United States Senate
candidate Harold Ford, Jr. He is a member of the Ford political family from Memphis.
</item 5>

<item 6>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 6>

<item 7>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 7>

<item 8>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 8>

<item 9>
L. R. Ford Jr.
Lester Randolph Ford Jr. (born September 23, 1927 – February 26, 2017) was an American mathematician specializing in network flow problems.

 He was the son of mathematician Lester R. Ford Sr.
</item 9>

<item 10>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 10>
=== END ITEM id=37d89418adeaf353 ===
