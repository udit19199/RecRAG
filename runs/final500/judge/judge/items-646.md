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

This file has 11 items.

=== ITEM id=edf7323c5835d94b ===
Gold sentences and facts:
Sentence 0: The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
  Fact 0.0: The Silent Historian has the original title 'Het zwijgen van Loe de Jong [The Silence of Loe de Jong]'.
  Fact 0.1: The Silent Historian is a 2011 Dutch documentary film.
  Fact 0.2: The Silent Historian is a documentary film by Simonka de Jong.
  Fact 0.3: The Silent Historian was shown as part of the January 2012 New York Jewish Film Festival.
  Fact 0.4: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Jewish Museum.
  Fact 0.5: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Film Society of Lincoln Center.
Sentence 1: The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
  Fact 1.0: The Betrayal — Nerakhoon is a 2008 documentary film.
  Fact 1.1: The Betrayal — Nerakhoon was directed by Ellen Kuras.
  Fact 1.2: The Betrayal — Nerakhoon was directed by Thavisouk Phrasavath.

Retrieved items:
<item 1>
Betrayal (1929 film)
Betrayal was a 1929 drama film produced for Famous Players-Lasky and released by Paramount Pictures.

 The film is the last silent film directed by Lewis Milestone, the last silent performance by Gary Cooper, the last silent performance by Germany's Emil
Jannings, and the only onscreen pairing of Cooper and Jannings.

 It is considered a lost film.
</item 1>

<item 2>
Graph query row:
work: The Silent Historian
genre: None
film_type: documentary film
genre_concept: None
</item 2>

<item 3>
The Decembrists (film)
The Decembrists (Russian: Декабристы , "Dekabristy " ) is a 1927 Soviet silent historical drama film directed by Aleksandr Ivanovsky.
</item 3>

<item 4>
The Price of Betrayal
The Price of Betrayal (Swedish: "Judaspengar" ) is a 1915 Swedish silent drama film directed by Victor Sjöström.
</item 4>

<item 5>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 5>

<item 6>
The Counterfeiters (2007 film)
The Counterfeiters (German: "Die Fälscher" ) is a 2007 Austrian-German drama film written and directed by Stefan Ruzowitzky.

 It fictionalizes Operation Bernhard, a secret plan by Nazi Germany during World War II to destabilize the United Kingdom by flooding its economy with forged
Bank of England pound notes.

 The film centres on a Jewish counterfeiter, Salomon 'Sally' Sorowitsch, who is coerced into assisting the operation at the Sachsenhausen concentration camp.
</item 6>

<item 7>
Graph query row:
work: The Betrayal – Nerakhoon
genre: None
film_type: documentary film
genre_concept: None
</item 7>

<item 8>
Thérèse Raquin (1928 film)
Thérèse Raquin is a 1928 drama film directed by Jacques Feyder.

 It is the third silent film adaptation of the novel of the same name by Émile Zola.

 The film stars Gina Manès as Thérèse Raquin, Wolfgang Zilzer as Monsieur Raquin, and Jeanne Marie Laurent as Madame Raquin.

 The décors of the Paris suburbs for the film were built by André Andrejew.

 The film was produced by Deutsche Film Union in Germany, with German and French actors, in a French-German co-production, to be later released at the same time
in France as "Thérèse Raquin" and Germany as Du sollst nicht ehebrechen!

.

 As no words were spoken, both versions differed only with the language of intertitles.

 The British title at the time of the film's original release was Thou Shalt Not.

 This is last silent film imports distributed by Warner Bros.' newly acquired First National subsidiary.

 No dialong with music score and sound effects.
</item 8>

<item 9>
The Betrayal – Nerakhoon
The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
</item 9>

<item 10>
The Silent Historian
The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as
part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
</item 10>
=== END ITEM id=edf7323c5835d94b ===

=== ITEM id=df02d4f44f8ae1b6 ===
Gold sentences and facts:
Sentence 0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area women role models over 85 years old who remained artistically active.
  Fact 0.0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a documentary film.
  Fact 0.1: Still Kicking: Six Artistic Women of Project Arts & Longevity was released in 2006.
  Fact 0.2: Still Kicking: Six Artistic Women of Project Arts & Longevity runs 32 minutes.
  Fact 0.3: Still Kicking: Six Artistic Women of Project Arts & Longevity was made by Pacific Grove filmmaker Greg Young.
  Fact 0.4: Still Kicking: Six Artistic Women of Project Arts & Longevity features six Bay Area women role models.
  Fact 0.5: The six women featured in Still Kicking are over 85 years old.
  Fact 0.6: The six women featured in Still Kicking remained artistically active.
Sentence 1: Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
  Fact 1.0: Time Bombs is a film.
  Fact 1.1: Time Bombs is a 2008 film.
  Fact 1.2: Time Bombs is a Canadian film.
  Fact 1.3: Time Bombs was produced by Productions de la ruelle.

Retrieved items:
<item 1>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
</item 1>

<item 2>
Guylaine Maroist
Guylaine Maroist is a Canadian journalist, filmmaker, musician, scriptwriter and film director.

 She is well known for her documentary productions such as "Gentilly or Not To Be", "Time Bombs", "The Disunited States of Canada", and "God Save Justin
Trudeau".

 In 2011 she received the Governor General’s History Award for Popular History (The Pierre Berton Award) for her TV documentary series "J’ai la mémoire qui
tourne" ("My Memories On a Roll").

 She is President of Productions de la Ruelle, a documentary film production company in Montreal, and President of Les Artistes pour la Paix, a Quebec NGO
advocating peace and nuclear disarmament.
</item 2>

<item 3>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 3>

<item 4>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 4>

<item 5>
Still Kicking: The Fabulous Palm Springs Follies
Still Kicking: The Fabulous Palm Springs Follies is a 1997 American short documentary film directed by Mel Damski.

 It was nominated at the 70th Academy Awards for Best Documentary Short Subject.

 It features The Fabulous Palm Springs Follies, which was formerly staged at the Palm Springs, California Plaza Theatre.
</item 5>

<item 6>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 6>

<item 7>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 7>

<item 8>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 8>

<item 9>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 9>

<item 10>
Time Bomb High School
Time Bomb High School is the second album by the Reigning Sound.

 It was released in 2002 on In the Red Records.

 The album featured the original Reigning Sound lineup of Greg Cartwright on lead vocals and guitar; Alex Greene on organ, piano, guitar, and backing vocals;
Jeremy Scott on bass, and backing vocals; and Greg Roberson on drums.

  Howlin' Pelle Almqvist of The Hives commented in "Rolling Stone" that Time Bomb High School was his favorite record of 2002.
</item 10>
=== END ITEM id=df02d4f44f8ae1b6 ===

=== ITEM id=b1cac661f5ac2951 ===
Gold sentences and facts:
Sentence 0: Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.
  Fact 0.0: Underdog is a 2007 American film.
  Fact 0.1: Underdog is a family superhero comedy film.
  Fact 0.2: Underdog is based on the 1960s cartoon series of the same name.
Sentence 1: Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi, a Life in the Woods" by Austrian author Felix Salten.
  Fact 1.0: Bambi is a 1942 American animated film.
  Fact 1.1: Bambi was directed by David Hand.
  Fact 1.2: David Hand supervised a team of sequence directors on Bambi.
  Fact 1.3: Bambi was produced by Walt Disney.
  Fact 1.4: Bambi is based on the book 'Bambi, a Life in the Woods'.
  Fact 1.5: The book 'Bambi, a Life in the Woods' was written by Austrian author Felix Salten.

Retrieved items:
<item 1>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 1>

<item 2>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 2>

<item 3>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 3>

<item 4>
Bosko the Musketeer
Bosko the Musketeer is an American animated short film.

 It is a "Looney Tunes" cartoon, featuring Bosko, the first star of the series.

 It was released on August 12, 1933, although some sources note September 16 as a date; this is problematic, as that would imply that the last films featuring
Bosko as the star of Warner Bros. cartoons were released after the first film featuring Buddy, the second star of the series.

 It was, like most "Looney Tunes" of the time, directed by Hugh Harman; Frank Marsales was the musical director.
</item 4>

<item 5>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 5>

<item 6>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 6>

<item 7>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
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
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 9>

<item 10>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 10>
=== END ITEM id=b1cac661f5ac2951 ===

=== ITEM id=81be136d80a41d8b ===
Gold sentences and facts:
Sentence 0: 'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.
  Fact 0.0: A Guarani word, written in the source as 'aba], means 'what gives strength to the Indian'.
  Fact 0.1: The Guarani word is used for the infusions of the bark of a number of trees native to Brazil.
Sentence 1: Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South America that belongs to the Tupi–Guarani family of the Tupian languages.
  Fact 1.0: Guarani specifically refers to the primary variety known as Paraguayan Guarani.
  Fact 1.1: Paraguayan Guarani has the endonym avañe'ẽ, meaning 'the people's language'.
  Fact 1.2: Guarani is an indigenous language of South America.
  Fact 1.3: Guarani belongs to the Tupi–Guarani family of the Tupian languages.

Retrieved items:
<item 1>
Achomawi
In anthropology, Achomawi (also Achumawi, Ajumawi and Ahjumawi) are the northerly nine (out of eleven) bands of the Pit River tribe of Native Americans who
lived in what is now northeastern California in the United States.

 These nine autonomous bands (also called "tribelets") of the Pit River Indians spoke various dialects of one common language, and the other two bands spoke
dialects of a related language, called Atsugewi.

 The name "Achomawi" means river dwelling (from ajumma, "river") and the band historically inhabited the Fall River Valley and the Pit River from the south end
of Big Valley Mountains, westerly to Pit River Falls.
</item 1>

<item 2>
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 2>

<item 3>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 3>

<item 4>
Guarani language
Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South
America that belongs to the Tupi–Guarani family of the Tupian languages.

 It is one of the official languages of Paraguay (along with Spanish), where it is spoken by the majority of the population, and where half of the rural
population is monolingual.

 It is spoken by communities in neighboring countries, including parts of northeastern Argentina, southeastern Bolivia and southwestern Brazil, and is a second
official language of the Argentine province of Corrientes since 2004; it is also an official language of Mercosur.
</item 4>

<item 5>
Pagode (music)
Pagode is a type of Brazilian country-folk traditional style of music.

 It's also known as Cipó Preto (black liana), Pagode caipira (rural pagode) or Pagode sertanejo (folk pagode).

 Sertanejo means anything that comes from the back-country, outback or simply the countryside.

 This style of music was pioneered by musicians Tião Carreiro and Lourival dos Santos in the late 1950s, when they fused the "Coco" and "Calango de roda"
rhythms.

 Although other instruments can be added, it is much characterized by its simple, acoustic approach, with the use of just one or two viola guitars, often
accompanied by an acoustic guitar and, at times, a light percussion section.

 A structural feature commonly used in this genre is the alternation of verses sung with the accompaniement of a classical guitar, with solos of viola caipira.

 Pieces can be entirely instrumental or have one or two singers, usually: when there are two lead vocals (that can be men, women or a combination), they sing
together on a steady interval.
</item 5>

<item 6>
Cora language
Cora is an indigenous language of Mexico of the Uto-Aztecan language family.

 It is spoken by the ethnic group that is widely known as the Cora but who refer to themselves as "Naáyarite".

 The Cora inhabit the northern sierra of the Mexican state Nayarit which is named after its indigenous inhabitants.

 Cora is a Mesoamerican language and shows many of the traits defining the Mesoamerican Linguistic Area.

 Under the General Law of Linguistic Rights of the Indigenous Peoples it is recognized as a "national language" along with 62 other indigenous languages and
Spanish which have the same "validity" in Mexico .
</item 6>

<item 7>
Catuaba
The name catuaba (pronounced [ka.two.

'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.

 The most widely used barks are derived from the trees "Trichilia catigua" and "Erythroxylum vaccinifolium".

 Other catuaba preparations use the bark of trees from the following genera or families: "Anemopaegma", "Ilex", "Micropholis", "Phyllanthus", "Secondatia",
"Tetragastris" and species from the Myrtaceae.
</item 7>

<item 8>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 8>

<item 9>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 9>

<item 10>
San Antonio River (South America)
The San Antonio River (Spanish, Río San Antonio, Portuguese, Rio Santo Antônio; also called San Antonio Guazú) is a tributary of the Iguazu River.

 The San Antonio River forms the border between Misiones Province in Argentina and Paraná State in Brazil.

 South of the San Antonio's source near Barracão, the international border continues south along the Pepiri-Guazu River, which forms the border between Misiones
Province and Santa Catarina State.
</item 10>
=== END ITEM id=81be136d80a41d8b ===

=== ITEM id=3645d990d25dc105 ===
Gold sentences and facts:
Sentence 0: San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San Antonio metropolitan area.
  Fact 0.0: San Antonio International Airport has the IATA code SAT.
  Fact 0.1: San Antonio International Airport has the ICAO code KSAT.
  Fact 0.2: San Antonio International Airport has the FAA LID SAT.
  Fact 0.3: San Antonio International Airport is an international airport.
  Fact 0.4: San Antonio International Airport is located in San Antonio, Texas.
  Fact 0.5: San Antonio International Airport serves the Greater San Antonio metropolitan area.
Sentence 1: Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.
  Fact 1.0: Yakutat Airport has the IATA code YAK.
  Fact 1.1: Yakutat Airport has the ICAO code PAYA.
  Fact 1.2: Yakutat Airport has the FAA LID YAK.
  Fact 1.3: Yakutat Airport is a state owned, public use airport.
  Fact 1.4: Yakutat Airport is located three nautical miles (6 km) southeast of the central business district of Yakutat.
  Fact 1.5: Yakutat is a city and borough in the U.S. state of Alaska.
  Fact 1.6: Yakutat has no road access to the outside world.

Retrieved items:
<item 1>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 1>

<item 2>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 2>

<item 3>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 3>

<item 4>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 4>

<item 5>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 5>

<item 6>
Stepanakert Airport
Stepanakert Airport (Armenian: Ստեփանակերտի Օդանավակայան ), is an airport in Khojaly, near Stepanakert, the capital city of the "de facto" independent Republic
of Artsakh (NKR) which is an internationally recognized part of Azerbaijan.

 The airport has been under the control of the Republic of Artsakh since 1992.

 Flights ceased to take place with the escalation of the Nagorno-Karabakh War in 1990.

 As the airport lies within the internationally recognized borders of Azerbaijan, and the current government is unrecognized, there are no codes for the airport
in the official IATA list.
</item 6>

<item 7>
Yakutat Airport
Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business
district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.

 Airline service is subsidized by the Essential Air Service program.
</item 7>

<item 8>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 8>

<item 9>
Kunming Changshui International Airport
Kunming Changshui International Airport (IATA: KMG, ICAO: ZPPP) is the primary airport serving Kunming, the capital of Yunnan Province, China.

 The airport is located 24.5 km northeast of the city center in a graded mountainous area about 2100 m above sea level.

 The airport opened at 08:00 (UTC+8) on June 28, 2012, replacing the old Kunming Wujiaba International Airport, which will be demolished.

 As a gateway to Southeast and South Asia, Changshui Airport is a hub for China Eastern Airlines, Kunming Airlines, Lucky Air, Sichuan Airlines and Ruili
Airlines.
</item 9>

<item 10>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 10>
=== END ITEM id=3645d990d25dc105 ===

=== ITEM id=020da690fe9ef2af ===
Gold sentences and facts:
Sentence 0: Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.
  Fact 0.0: Triumph of the Will's German title is Triumph des Willens.
  Fact 0.1: Triumph of the Will is a 1935 German propaganda film.
  Fact 0.2: Leni Riefenstahl directed Triumph of the Will.
  Fact 0.3: Leni Riefenstahl produced Triumph of the Will.
  Fact 0.4: Leni Riefenstahl edited Triumph of the Will.
  Fact 0.5: Leni Riefenstahl co-wrote Triumph of the Will.
Sentence 1: The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".
  Fact 1.0: The End of America is a 2008 documentary film.
  Fact 1.1: Annie Sundberg directed The End of America.
  Fact 1.2: Ricki Stern directed The End of America.
  Fact 1.3: The End of America is adapted from Naomi Wolf's 2007 book.

Retrieved items:
<item 1>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 1>

<item 2>
The End of America (film)
The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".

 The film analyzes policy changes made during the Bush Administration, and makes the case that these changes threaten American democracy.
</item 2>

<item 3>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 3>

<item 4>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 4>

<item 5>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 5>

<item 6>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 6>

<item 7>
Sarajevo (1940 French film)
Sarajevo (French:De Mayerling à Sarajevo) is a 1940 French historical film directed by Max Ophüls and starring Edwige Feuillère, John Lodge and Aimé Clariond.

 Beginning in the aftermath of the Mayerling Incident the film portrays the love affair and marriage between Archduke Franz Ferdinand of Austria and Sophie,
Duchess of Hohenberg, leading up to their eventual assassination in 1914 in events that triggered the First World War.

 The film was not a commercial or critical success.

 Following the German occupation of France the film was banned, and Ophüls fled into exile for the second time.
</item 7>

<item 8>
Empire (1964 film)
Empire is a 1964 American black and white silent film written, produced, and directed by Andy Warhol.

 It consists of eight hours and five minutes of continuous slow motion footage of the Empire State Building in New York City.

 Abridged showings of the film were never allowed, and supposedly the unwatchability of the film was an important part of the reason the film was created.

 However, a legitimate Italian VHS produced in association with the Andy Warhol Museum in 2000 contains only an extract of 60 minutes.

 Its use of the long take "in extremis" is an extension of Warhol's earlier work the previous year with "Sleep".

 Warhol employed Rob Trains to be the projectionist for a screening of the film.

 Trains miscalculated and mixed the order and speed of the reels for the eight-hour movie.

 After a positive review in "The New York Times", Warhol actually liked the "mistake" and employed Trains for the entire summer.
</item 8>

<item 9>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 9>

<item 10>
Triumph of the Will
Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.

 It chronicles the 1934 Nazi Party Congress in Nuremberg, which was attended by more than 700,000 Nazi supporters.

 The film contains excerpts from speeches given by Nazi leaders at the Congress, including Adolf Hitler, Rudolf Hess and Julius Streicher, interspersed with
footage of massed Sturmabteilung (SA) and Schutzstaffel (SS) troops and public reaction.

 Hitler commissioned the film and served as an unofficial executive producer; his name appears in the opening titles.

 The film's overriding theme is the return of Germany as a great power, with Hitler as the leader who will bring glory to the nation.

 Because the film was made after the 1934 Night of the Long Knives (on 30 June) many prominent Sturmabteilung (SA) members are absent—they were murdered in that
Party purge, organised and orchestrated by Hitler to replace the SA with the Schutzstaffel (SS) as his main paramilitary force.
</item 10>
=== END ITEM id=020da690fe9ef2af ===

=== ITEM id=3af5044f6007524c ===
Gold sentences and facts:
Sentence 0: Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
  Fact 0.0: Martin Archer Flavin was born on November 2, 1883.
  Fact 0.1: Martin Archer Flavin died on December 27, 1967.
  Fact 0.2: Martin Archer Flavin was an American playwright.
  Fact 0.3: Martin Archer Flavin was an American novelist.
Sentence 1: Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.
  Fact 1.0: Peter Bradford Benchley was born on May 8, 1940.
  Fact 1.1: Peter Bradford Benchley died on February 11, 2006.
  Fact 1.2: Peter Bradford Benchley was an American author.
  Fact 1.3: Peter Bradford Benchley was an American screenwriter.

Retrieved items:
<item 1>
Michael Breed
Michael S. Breed (born May 14, 1962, in Greenwich, Connecticut) is a professional golf instructor and television host.

 In 2003, he was selected as a Top 100 Instructor in America by "Golf Magazine", in 2011, he was voted one of the Top 50 Instructors in America by Golf Digest
(now 13th on this list, and #1 in the state of New York.)

, in 2012, Breed was chosen as the PGA's National Teacher of the Year.

 Prior to his time as a television host, Breed was the Head Golf Professional at Sunningdale Country Club for 12 years from 2001–2012, and the Head Golf
Professional at Birchwood Country Club, as well as the Assistant Professional at Deepdale Golf Club and Augusta National Golf Club.

 He has served on different boards for the Metropolitan PGA for over 10 years and has represented the PGA of America at the National Golf Day in Washington D.C.
since 2012.

 He is involved in charities such as The First Tee, Folds of Honor, Hope for the Warriors, Wounded Warriors Foundation and Salute Military Golf Association.
</item 1>

<item 2>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 2>

<item 3>
Peter Benchley
Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.

 He is known for the bestselling novel "Jaws" and co-wrote its subsequent film adaptation with Carl Gottlieb.

 Several more of his works were also adapted for cinema, including "The Deep", "The Island", "Beast", and "White Shark".
</item 3>

<item 4>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 4>

<item 5>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 5>

<item 6>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 6>

<item 7>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 7>

<item 8>
Martin Flavin (politician)
Martin Flavin (1841– 30 December 1916) was an Irish nationalist politician, butter merchant and prominent businessman from Cork.

 He was a Member of Parliament (MP) from 1891 to 1892 .
</item 8>

<item 9>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 9>

<item 10>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 10>
=== END ITEM id=3af5044f6007524c ===

=== ITEM id=99fb94e0ff5f6603 ===
Gold sentences and facts:
Sentence 0:  It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
  Fact 0.0: The area occupies 40 km (15 sq mi).
  Fact 0.1: The area extends into each of the six municipalities on the island.
Sentence 1: Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the fifth largest in Spain's national parks system.
  Fact 1.0: Sierra de Guadarrama National Park is a national park in Spain.
  Fact 1.1: The Spanish name of Sierra de Guadarrama National Park is Parque Nacional de la Sierra de Guadarrama.
  Fact 1.2: Sierra de Guadarrama National Park covers nearly 34,000 hectares.
  Fact 1.3: Sierra de Guadarrama National Park is the fifth largest in Spain's national parks system.

Retrieved items:
<item 1>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 1>

<item 2>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 2>

<item 3>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 3>

<item 4>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 4>

<item 5>
Golf La Moraleja
Golf La Moraleja is one of Spain’s most important clubs.

 It is the largest Spanish golf club for its number of its holes and the largest in Europe located in a metropolitan area.

 Founded in 1973, it currently has four 18-hole golf courses, all rated par 72, designed by Jack Nicklaus.

 It also contains a 9-hole short course, 8 tennis courts, 12 paddle tennis courts, 2 squash courts, one covered and three open-air swimming pools, a gym and a
spa.

 Added to these are two clubhouses, a children’s chalet and a tennis clubhouse (on 1 and 2 courses), plus a prebuilt modular clubhouse for courses 3 and 4.

 Courses 1 and 2 are located at La Moraleja residential estate (Alcobendas, Madrid).

 Courses 3 and 4 are located in the municipality of Algete.

 It has 6,000 shareholder members.
</item 5>

<item 6>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 6>

<item 7>
Golf La Moraleja
It has 6,000 shareholder members.

 In 2013 it was voted the best golf course in Spain by Deporte & Business and in 2014, La Moraleja 3 was voted the second best golf course in Spain by the
American magazine Golf Digest.

 The Club has hosted competitions such as the World Cup of Golf, the Spanish Open or the Paddle Tennis World Championship.
</item 7>

<item 8>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 8>

<item 9>
Pirin National Park
Pirin National Park (Bulgarian: Национален парк "Пирин" ) is a national park that encompasses the larger part of the Pirin Mountains in south-west Bulgaria,
spanning an area of 403.56 km2 .

 It is one of the three national parks in the country, the others being Rila National Park and Central Balkan National Park.

 The park was established in 1962 and its territory was expanded several times since then.

 Pirin National Park was declared a UNESCO World Heritage Site in 1983.

 The altitude varies from 950 m to 2,914 m at Vihren, Bulgaria's second highest summit and the Balkans' third.
</item 9>

<item 10>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 10>
=== END ITEM id=99fb94e0ff5f6603 ===

=== ITEM id=68ff8a1f059ce10d ===
Gold sentences and facts:
Sentence 0: The Great War is a 26-episode documentary series from 1964 on the First World War.
  Fact 0.0: The Great War is a documentary series.
  Fact 0.1: The Great War has 26 episodes.
  Fact 0.2: The Great War is a documentary series from 1964.
  Fact 0.3: The Great War is about the First World War.
Sentence 1:  The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian Broadcasting Commission.
  Fact 1.0: The Great War was a co-production of the Imperial War Museum.
  Fact 1.1: The Great War was a co-production of the British Broadcasting Corporation.
  Fact 1.2: The Great War was a co-production of the Canadian Broadcasting Corporation.
  Fact 1.3: The Great War was a co-production of the Australian Broadcasting Commission.
Sentence 2:  The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.
  Fact 2.0: Michael Redgrave was the narrator of The Great War.
  Fact 2.1: Marius Goring gave readings in The Great War.
  Fact 2.2: Ralph Richardson gave readings in The Great War.
  Fact 2.3: Cyril Luckham gave readings in The Great War.
  Fact 2.4: Sebastian Shaw gave readings in The Great War.
  Fact 2.5: Emlyn Williams gave readings in The Great War.
Sentence 3: Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.
  Fact 3.0: Imperial War Museums (IWM) is a British national museum organisation.
  Fact 3.1: Imperial War Museums has branches at five locations in England.
  Fact 3.2: Three of the five Imperial War Museums branches are in London.

Retrieved items:
<item 1>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 1>

<item 2>
Norman Gilliland
Norman Gilliland has been a producer on Wisconsin Public Radio since 1984, where he hosts classical music broadcasts, produces the interview program "University
of the Air," and reads for "Chapter A Day."

 He holds degrees in English and Broadcasting from the University of Florida and attended graduate school in English at Duke University, where he developed an
interest in broadcasting.

 He is also an active author with four published books, the historical novel "Sand Mansions" and its stand-alone sequel "Midnight Catch," "Downeast Ledge"
(2013), plus two nonfiction books about classical music--"Grace Notes for a Year" and "Scores to Settle."

 He has produced an audio drama based upon Dick Ringler's modern English translation of the Old English narrative Beowulf titled "Beowulf: The Complete Story—A
Drama" (ISBN  ).

 He was one of a handful of experts interviewed in the Academy Award winning short documentary .
</item 2>

<item 3>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 3>

<item 4>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 4>

<item 5>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 5>

<item 6>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 6>

<item 7>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 7>

<item 8>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 8>

<item 9>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 9>

<item 10>
First Life (TV series)
First Life is a 2010 British nature documentary series written and presented by David Attenborough, also known by the expanded titles David Attenborough's First
Life (UK) and First Life with David Attenborough (USA).

 It was first broadcast in the USA as a two-hour special on the Discovery Channel on 24 October 2010.

 In the United Kingdom it was broadcast as a two-part series on BBC Two on 5 November 2010.

 "First Life" sees Attenborough tackle the subject of the origin of life on Earth.

 He investigates the evidence from the earliest fossils, which suggest that complex animals first appeared in the oceans around 540 million years ago, an event
known as the Cambrian Explosion.

 Trace fossils of multicellular organisms from an even earlier period, the Ediacaran biota, are also examined.

 Attenborough travels to Canada, Morocco and Australia, using some of the latest fossil discoveries and their nearest equivalents amongst living species to
reveal what life may have been like at that time.

 Visual effects and computer animation are used to reconstruct and animate the extinct life forms.
</item 10>
=== END ITEM id=68ff8a1f059ce10d ===

=== ITEM id=ed36333337cae6b1 ===
Gold sentences and facts:
Sentence 0: "Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".
  Fact 0.0: 'Empty Orchestra' is the fourth episode of the third series of Inside No. 9.
  Fact 0.1: Inside No. 9 is a British dark comedy anthology television programme.
Sentence 1:  Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.
  Fact 1.0: 'Empty Orchestra' was written by Steve Pemberton.
  Fact 1.1: 'Empty Orchestra' was written by Reece Shearsmith.
  Fact 1.2: 'Empty Orchestra' was directed by Guillem Morales.
  Fact 1.3: 'Empty Orchestra' first aired on 7 March 2017.
  Fact 1.4: 'Empty Orchestra' first aired on BBC 2.
Sentence 2:  The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).
  Fact 2.0: The episode 'Empty Orchestra' is set in a karaoke booth.
  Fact 2.1: The episode 'Empty Orchestra' follows a group of work colleagues.
  Fact 2.2: The work colleague Greg is shown as Shearsmith.
  Fact 2.3: The work colleague Fran is shown as Sarah Hadland.
  Fact 2.4: The work colleague Connie is shown as Tamzin Outhwaite.
  Fact 2.5: The work colleague Janet is shown as Emily Howlett.
  Fact 2.6: The work colleague Duane is shown as Javone Prince.
  Fact 2.7: The colleagues are celebrating the promotion of Roger.
  Fact 2.8: Roger is shown as Pemberton.
Sentence 3:  Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and television productions, including army series "Red Cap" and crime drama "New Tricks".
  Fact 3.0: The performer described in this sentence came to national notice playing Melanie Owen in the BBC One soap opera EastEnders from 1998 until 2002.
  Fact 3.1: The performer has since starred in theatre productions.
  Fact 3.2: The performer has since starred in television productions.
  Fact 3.3: The performer has starred in the army series Red Cap.
  Fact 3.4: The performer has starred in the crime drama New Tricks.

Retrieved items:
<item 1>
Empty Orchestra
"Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".

 Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.

 The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily
Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).

 Rebekah Hinds also stars.
</item 1>

<item 2>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 2>

<item 3>
Melanie Owen
Melanie Jane "Mel" Owen (also Healy and Beale) is a fictional character from the BBC soap opera "EastEnders", played by Tamzin Outhwaite.

 She appeared from 19 October 1998 to 12 April 2002 when Outhwaite opted to leave the role.

 Mel was introduced by executive producer Matthew Robinson and became a prominent character during her tenure in the soap.
</item 3>

<item 4>
The entries profile actors and television productions. Tommy Knight is known for The Sarah Jane Adventures and Doctor Who; Mark Benton played Daniel Chalk in
Waterloo Road; Robin Chalk appeared in Dirty Dancing and Moon. Tamzin Outhwaite starred in EastEnders, Red Cap, Paradox, Vital Signs and Inside No. 9’s “Empty
Orchestra.” Dominic West is known for The Wire and Appropriate Adult; John Altman played Nick Cotton in EastEnders. Other entries cover children’s
</item 4>

<item 5>
Tamzin Outhwaite
Tamzin Maria Outhwaite ( ; born 5 November 1970) is a British actress from London.

 Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and
television productions, including army series "Red Cap" and crime drama "New Tricks".
</item 5>

<item 6>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 6>

<item 7>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 7>

<item 8>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 8>

<item 9>
The entries profile performers and productions, including *Doctor Who*, *EastEnders* and *Waterloo Road*. Juliet Weber is a producer, director, writer and
editor. Juno Temple starred in *Vinyl*; Holly Hunter won major awards for *The Piano*. Kathryn Drysdale played Louise Brooks and Meghan Markle. Kate Winslet
rose to prominence with *Heavenly Creatures* and *Titanic*. Cate Blanchett won two Oscars, including for *The Aviator*. Reese
</item 9>

<item 10>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 10>
=== END ITEM id=ed36333337cae6b1 ===

=== ITEM id=8d7cf6c01782d824 ===
Gold sentences and facts:
Sentence 0:  It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.
  Fact 0.0: The car described in this sentence was driven by Eddie Irvine.
  Fact 0.1: Eddie Irvine was in his second year with the team.
  Fact 0.2: The car described in this sentence was also driven by Luciano Burti.
  Fact 0.3: Luciano Burti had deputised for Eddie Irvine at the 2000 Austrian Grand Prix.
Sentence 1: Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.
  Fact 1.0: Edmund Irvine Jr. was born on 10 November 1965.
  Fact 1.1: Edmund Irvine Jr. is a retired British racing driver.
  Fact 1.2: Edmund Irvine Jr. is from Northern Ireland.

Retrieved items:
<item 1>
1999 British Grand Prix
The 1999 British Grand Prix was a Formula One motor race held on 11 July 1999 at the Silverstone Circuit near Silverstone, England.

 It was the eighth race of the 1999 Formula One season.

 The 60-lap race was won by McLaren driver David Coulthard after he started from third position.

 Eddie Irvine finished second for the Ferrari team and Williams driver Ralf Schumacher came in third.
</item 1>

<item 2>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 2>

<item 3>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 3>

<item 4>
2001 Monaco Grand Prix
The 2001 Monaco Grand Prix (formally the LIX Grand Prix de Monaco) was a Formula One motor race held at the Circuit de Monaco on 27 May 2001.

 It was the seventh race of the 2001 Formula One season.

 The 78-lap race was won by Michael Schumacher driving for the Ferrari team.

 Rubens Barrichello finished second in the other Ferrari with Eddie Irvine third for the Jaguar team.

 Schumacher's win was his fourth of season, and Irvine's third place was the first podium position for the Jaguar team.
</item 4>

<item 5>
1997 Japanese Grand Prix
The 1997 Japanese Grand Prix (officially known as the XXIII Fuji Television Japanese Grand Prix) was a Formula One motor race held on 12 October 1997 at the
Suzuka Circuit, Suzuka.

 It was the 16th and penultimate race of the 1997 Formula One season.

 The 53-lap race was won by Michael Schumacher for the Ferrari team after starting from second position.

 Heinz-Harald Frentzen finished second in a Williams, and Eddie Irvine third in the other Ferrari.

 Irvine led much of the race before moving over to assist Schumacher's championship battle by blocking Drivers' Championship leader Jacques Villeneuve.
</item 5>

<item 6>
Jordan 194
The Jordan 194 was a Formula One car for the 1994 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 Kelvin Burt was named as test driver, but his mileage in the car was limited.

 The engine was a Hart 1035 3.5 V10, a developed version of the engine which had proved promising in 1993.

 The team's main sponsor was Sasol, with additional financial support coming from Arisco, the Irish government and other smaller sponsors.

 With driver aids such as traction control and active suspension banned for the 1994 season, the car used a conventional semi-automatic gearbox and pushrod
double wishbone suspension.

 The simpler technical regulations of 1994 seemed to benefit Jordan and the 194 recaptured the level of performance seen in the team's debut season, 1991.

 Indeed, the cut-down airbox and drooping front wing of the 194 recalled the design features of the 1991 car.
</item 6>

<item 7>
1997 Argentine Grand Prix
The 1997 Argentine Grand Prix was a Formula One motor race held at Autódromo Oscar Alfredo Gálvez in Buenos Aires, Argentina on 13 April 1997.

 It was the third race of the 1997 Formula One season and the 600th World Championship event.

 The 72-lap race was won by Williams driver Jacques Villeneuve after he started from pole position.

 Eddie Irvine finished second for the Ferrari team and Jordan driver Ralf Schumacher came in third.
</item 7>

<item 8>
1999 Japanese Grand Prix
The 1999 Japanese Grand Prix (formally the XXV Fuji Television Japanese Grand Prix) was a Formula One motor race held on 31 October 1999 at the Suzuka
International Racing Course in Suzuka, Japan.

 It was the sixteenth and final round of the 1999 Formula One season.

 The 53-lap race was won by McLaren driver Mika Häkkinen after starting from second position.

 Michael Schumacher finished second in a Ferrari with team-mate Eddie Irvine finishing third.

 Häkkinen's victory confirmed him as 1999 Drivers' Champion.

 Ferrari were also confirmed as Constructors' Champions.
</item 8>

<item 9>
Jaguar R2
The Jaguar R2 was the car with which the Jaguar team competed in the Formula One season.

 It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.

 However, the Brazilian was replaced by Pedro de la Rosa after only four Grands Prix.
</item 9>

<item 10>
Eddie Irvine
Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.

 He was a Formula One driver between 1993 and 2002, and runner-up in the 1999 World Drivers' Championship, driving for Scuderia Ferrari.
</item 10>
=== END ITEM id=8d7cf6c01782d824 ===
