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

=== ITEM id=cff21d9913b1f4de ===
Gold sentences and facts:
Sentence 0: Catcha Lake is a Canadian lake located in the central part of Nova Scotia's Halifax Regional Municipality.
  Fact 0.0: Catcha Lake is a Canadian lake
  Fact 0.1: Catcha Lake is located in the central part of Nova Scotia's Halifax Regional Municipality
Sentence 1: The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.
  Fact 1.0: The Hedley Mascot Mine was a gold mine
  Fact 1.1: The Hedley Mascot Mine was in Hedley, British Columbia, Canada

Retrieved items:
<item 1>
Hedley, British Columbia
Hedley is an unincorporated town in southern British Columbia, Canada, named after Robert R. Hedley, the manager of the Hall Smelter in Nelson.

 Hedley is located at the foot of Nickel Plate Mountain in the Similkameen.

 The town had a population of approximately 400 as of 2005.

 In the early 1900s, Hedley's population peaked over 1,000 people, primarily due to the gold mining industry.

 The Hedley Museum and the Mascot Mine Museum display artifacts and photographs from this era.
</item 1>

<item 2>
Catcha Lake
Catcha Lake is a Canadian lake located in the central part of Nova Scotia's Halifax Regional Municipality.
</item 2>

<item 3>
Electoral district of Bass
The Electoral district of Bass is an electoral district of the Victorian Legislative Assembly.

 It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip
Island.

 It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi,
and part of suburban Pakenham.

 It lies within the Eastern Victoria Region of the upper house, the Legislative Council.
</item 3>

<item 4>
Victor Diamond Mine
The Victor Diamond Mine is the first Canadian diamond mine located in Ontario, and De Beers' second diamond mine in Canada (after the Snap Lake Diamond Mine).

 It is located in the Northern Ontario Ring of Fire, in the James Bay Lowlands 90 km west of Attawapiskat in the remote northern part of the province.

 In June 2005, the Attawapiskat First Nation voted in favour (85.5%) of ratifying the Impact Benefit Agreement (IBA).

 Construction of the mine began in February 2006 which created 3200 positions; mining and operations will create around 400 permanent positions.

 The Victor Mine is an open-pit mine, with a processing plant, workshops, and an airstrip located on site.

 By 2013-2014 royalties collected from De Beers Victor Diamond Mine amounted to $226.

 At that time De Beers was continuing to pay off its "$1 billion investment to build the mine and from now until it closes, the company expects to pay tens of
millions of dollars in royalties."
</item 4>

<item 5>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 5>

<item 6>
Hounsfield Heights/Briar Hill, Calgary
Hounsfield Heights/Briar Hill is an inner suburban neighbourhood in northwest Calgary, Alberta, Canada.

 Located north of the Hillhurst and West Hillhurst communities, the boundaries of the district are 16th Avenue N (Trans-Canada Highway)to the north; 14th Street
W to the east; Lane north of 7th Avenue N to 19th Street W and 8th Avenue N to the south; and Crowchild Trail, 12th Avenue N, Juniper Road, and 22nd Street W to
the west.

 Lions Park C-Train station is located within the community.

 The community is built on an escarpment and is popular for its views of downtown to the south and the Rocky Mountains to the west.
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
Hedley Mascot Mine
The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.

 Gold was first discovered in the Nickel Plate Mountain area in 1897 and several small mines were developed over the years.

 The Hedley Mascot Mine operated between 1936 and 1949 and was one of the most unusual mining operations in the world, being built entirely on the side of a
mountain, 5,000 feet above the town of Hedley or seven thousand feet above sea level.

 In the 1990s, the British Columbia government was going to burn the site down because it posed a safety risk, but the Minister of Tourism at the time
intervened and, in 1995, steps were taken to preserve the site as a Provincial Heritage resource.
</item 8>

<item 9>
Alpine National Park (Victoria): 646,000 ha, Australia’s largest; includes Mount Bogong (1,986 m), borders Kosciuszko NP, and joined the Australian National
Heritage List in 2008. Tweed Volcano: 23-million-year-old NSW shield volcano; erosion formed the Southern Hemisphere’s largest caldera. Barry Way: remote 170-km
alpine route from Jindabyne to Buchan, partly unsealed and seasonally hazardous. Broken Hill
</item 9>

<item 10>
Lake Shore Mine
The Lake Shore Mine is a gold mine located in Kirkland Lake, Ontario.

 In July 1912, Harry Oakes staked claims L-2605-6 which were in the lake itself and had reverted for non-performance of work.

 On September 6, 1912, he registered the transfer of claim L-1557 that Melville McDougall had staked for Oakes previously.

 On September 23, 1911, Harry Oakes registered the transfer of claim T-16635 from George Minaker and named the property Lake Shore.

 In production from 1918 to 1965, the mine produced almost 8.5 million ounces of gold and represented over a third of the gold produced in the entire camp.
</item 10>
=== END ITEM id=cff21d9913b1f4de ===

=== ITEM id=51370f26f0302ac6 ===
Gold sentences and facts:
Sentence 0: Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.
  Fact 0.0: Bluebeard's Castle is a one-act opera.
  Fact 0.1: Bluebeard's Castle is by Hungarian composer Béla Bartók.
  Fact 0.2: The Hungarian title of Bluebeard's Castle is A kékszakállú herceg vára.
  Fact 0.3: Literally, the Hungarian title means The Blue-Bearded Duke's Castle.
Sentence 1: Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.
  Fact 1.0: Agrippina (HWV 6) is an opera seria.
  Fact 1.1: Agrippina (HWV 6) has three acts.
  Fact 1.2: Agrippina (HWV 6) is by George Frideric Handel.
  Fact 1.3: Agrippina (HWV 6) has a libretto by Cardinal Vincenzo Grimani.

Retrieved items:
<item 1>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 1>

<item 2>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 2>

<item 3>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 3>

<item 4>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 4>

<item 5>
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 5>

<item 6>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 6>

<item 7>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
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
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 9>

<item 10>
The Padlock
The Padlock is a two-act 'afterpiece' opera by Charles Dibdin.

 The text was by Isaac Bickerstaffe.

 It debuted in 1768 at the Drury Lane Theatre in London as a companion piece to "The Earl of Warwick".

 It partnered other plays before a run of six performances in tandem with "The Fatal Discovery" by John Home.

 "The Padlock" was a success, largely due to Dibdin's portrayal of Mungo, a blackface caricature of a black servant from the West Indies.

 The company took the production to the United States the next year, where a portrayal by Lewis Hallam, Jr. as Mungo met with even greater accolades.

 The libretto was first published in London in 1768 (?)

 and in Dublin in 1775.

 The play remained in regular circulation in the U.S. as late as 1843.

 It was revived by the Old Vic Company in London and on tour in the UK in 1979 in a new orchestration by Don Fraser and played in a double-bill with Garrick's
"Miss In Her Teens".
</item 10>
=== END ITEM id=51370f26f0302ac6 ===

=== ITEM id=8ae2ad9604cd0779 ===
Gold sentences and facts:
Sentence 0: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 0.0: Henry Louis Mencken lived from September 12, 1880 to January 29, 1956.
  Fact 0.1: Henry Louis Mencken was an American journalist.
  Fact 0.2: Henry Louis Mencken was an American satirist.
  Fact 0.3: Henry Louis Mencken was an American cultural critic.
  Fact 0.4: Henry Louis Mencken was a scholar of American English.
Sentence 1: Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.
  Fact 1.0: Albert Camus lived from 7 November 1913 to 4 January 1960.
  Fact 1.1: Albert Camus was a French philosopher.
  Fact 1.2: Albert Camus was an author.
  Fact 1.3: Albert Camus was a journalist.

Retrieved items:
<item 1>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 1>

<item 2>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 2>

<item 3>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 3>

<item 4>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 4>

<item 5>
Reflections on the Guillotine
"Reflections on the Guillotine" is an extended essay written in 1957 by Albert Camus.

 In the essay Camus takes an uncompromising position for the abolition of the death penalty.

 Camus's view is similar to that of Cesare Beccaria and the Marquis de Sade, the latter having also argued that murder premeditated and carried out by the state
was the worst kind.

 Camus states that he does not base his argument on sympathy for the convicted but on logical grounds and on proven statistics.

 Camus also argues that capital punishment is an easy option for the government where remedy and reform may be possible.
</item 5>

<item 6>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 6>

<item 7>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 7>

<item 8>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 8>

<item 9>
The Guest
"The Guest" (French: "L'Hôte" ) is a short story by the French writer Albert Camus.

 It was first published in 1957 as part of a collection entitled "Exile and the Kingdom" ("L'exil et le royaume").

 The French title "L'Hôte" translates into both "the guest" and "the host" which ties back to the relationship between the main characters of the story.

 Camus employs this short tale to reflect upon issues raised by the political situation in French North Africa.

 In particular, he explores the problem of refusing to take sides in the colonial conflict in Algeria, something that mirrors Camus' own non-aligned stance
which he had set out in his Nobel Prize acceptance speech.
</item 9>

<item 10>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 10>
=== END ITEM id=8ae2ad9604cd0779 ===

=== ITEM id=7dd30628ea127f2f ===
Gold sentences and facts:
Sentence 0: William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.
  Fact 0.0: William Dwight Schultz was born on November 24, 1947.
  Fact 0.1: William Dwight Schultz is an American actor.
  Fact 0.2: William Dwight Schultz is an American voice artist.
Sentence 1:  He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the Squirrel in "CatDog".
  Fact 1.0: William Dwight Schultz is well known in animation as the mad scientist Dr. Animo in the "Ben 10" series.
  Fact 1.1: William Dwight Schultz is well known in animation as Chef Mung Daal in the children's cartoon "Chowder".
  Fact 1.2: William Dwight Schultz is well known in animation as Eddie the Squirrel in "CatDog".
Sentence 2: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 2.0: Chowder is an American animated television series.
  Fact 2.1: Chowder was created by C. H. Greenblatt.
  Fact 2.2: Chowder was created for Cartoon Network.

Retrieved items:
<item 1>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 1>

<item 2>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 2>

<item 3>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 3>

<item 4>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 4>

<item 5>
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
</item 5>

<item 6>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 6>

<item 7>
Harvey Beaks
Harvey Beaks is an American animated television series created by C.H. Greenblatt for Nickelodeon.

 The series premiered on March 28, 2015 following the 2015 Kids' Choice Awards.
</item 7>

<item 8>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 8>

<item 9>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 9>

<item 10>
List of Harvey Beaks episodes
"Harvey Beaks" is an American animated television series created by C. H. Greenblatt for Nickelodeon that first aired on March 28, 2015.
</item 10>
=== END ITEM id=7dd30628ea127f2f ===

=== ITEM id=66b59a3f3b310791 ===
Gold sentences and facts:
Sentence 0:  It is found in Yemen and India.
  Fact 0.0: The subject of this sentence is found in Yemen.
  Fact 0.1: The subject of this sentence is found in India.
Sentence 1:  Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.
  Fact 1.0: Yemen's capital has been temporarily relocated to the port city of Aden.
  Fact 1.1: Aden is on the southern coast.
  Fact 1.2: The relocation is stated to be because of the content of the previous sentence.

Retrieved items:
<item 1>
Al Khor (city)
Al Khor is a coastal city in northern Qatar, located 50 kilometres north of the capital, Doha.

 It is the capital city of the municipality of Al Khor.

 The name of the city means creek in Arabic as the town is located on a creek.

 Al Khor is home to many employees of the oil industry due to its proximity to Qatar's northern oil and natural gas fields, and due to its proximity to the Ras
Laffan Industrial City.
</item 1>

<item 2>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 2>

<item 3>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 3>

<item 4>
Ornithodoros savignyi
Because of its habit of feeding and dropping from its host, adult dispersal is limited, whereas larvae may remain attached to their hosts for several days.

 During its life cycle it will feed on multiple hosts between moults.
</item 4>

<item 5>
Amka
Amka (Hebrew: עַמְקָה‎ ), also known in Arabic as Amqa (Arabic: عمقا‎ ‎ ), is a moshav in the Matte Asher Regional Council of Israel's Northern District, near
Acre.

 The location of the moshav roughly corresponds the former Arab village, depopulated during the 1948 Arab–Israeli War.

 Yemenite Jews, who arrived from the southern Arab country of Yemen, founded the village's successor Amka in 1949.

 In 2016 its population was 706 .
</item 5>

<item 6>
Aulacodes templalis
Aulacodes templalis is a moth in the Crambidae family.

 It was described by Schaus in 1906.

 It is found in Brazil.
</item 6>

<item 7>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 7>

<item 8>
Yemen
Yemen ( ; Arabic: اليَمَن‎ ‎ "al-Yaman "), officially known as the Republic of Yemen (الجمهورية اليمنية "al-Jumhūrīyah al-Yamanīyah "), is an Arab country in
Western Asia at the southern end of the Arabian Peninsula.

 Yemen is the second-largest country in the peninsula, occupying 527,970 km (203,850 sq mi).

 The coastline stretches for about 2,000 km (1,200 mi).

 It is bordered by Saudi Arabia to the north, the Red Sea to the west, the Gulf of Aden and Arabian Sea to the south, and Oman to the east-northeast.

 Although Yemen's constitutionally stated capital is the city of Sana'a, the city has been under rebel control since February 2015.

 Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.

 Yemen's territory includes more than 200 islands; the largest of these is Socotra.
</item 8>

<item 9>
Agrotis orthogonia
The Pale Western Cutworm ("Agrotis orthogonia") is a moth of the Noctuidae family.

 It is found in North America, more specifically dry, semi-desert areas of western North America from southern Canada to California, ranging eastward nearly to
the eastern edge of the Great Plains.
</item 9>

<item 10>
Ornithodoros savignyi
Ornithodoros savignyi aka Sand tampan or African eyed tampan or Kalahari sand tampan, is one of some 37 species in the genus "Ornithodoros" and is a soft tick
with a leathery, mammillated integument, causing paralysis and tampan toxicosis, two unrelated conditions.

 The sand tampan is an ectoparasite on humans, their livestock and wild animals, including birds and bats.

 Occurring in semi-desert areas of Africa, Saudi Arabia and other parts of the Persian Gulf, India, Sri Lanka and into Asia, it is able to survive for lengthy
periods without feeding, spending most of its life burrowed under sand or loose soil, often in wait for animals that rest or sleep under trees or in the lee of
rocks, but also in places where people or their animals congregate such as marketplaces, places of worship, cattle kraals and village squares.

 The timing of its activity is geared to coincide with that of potential hosts, but hot sunny conditions are usually avoided.
</item 10>
=== END ITEM id=66b59a3f3b310791 ===

=== ITEM id=e1e5dc593d74da46 ===
Gold sentences and facts:
Sentence 0: John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.
  Fact 0.0: John Kingsley Orton, known as Joe Orton, was born on 1 January 1933.
  Fact 0.1: Joe Orton died on 9 August 1967.
  Fact 0.2: Joe Orton was an English playwright.
  Fact 0.3: Joe Orton was an English author.
Sentence 1:  His public career was short but prolific, lasting from 1964 until his death three years later.
  Fact 1.0: Joe Orton's public career was short but prolific.
  Fact 1.1: Joe Orton's public career lasted from 1964 until his death.
  Fact 1.2: Joe Orton died three years after his public career began in 1964.
Sentence 2:  During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.
  Fact 2.0: During his brief public career, Joe Orton shocked audiences with his scandalous black comedies.
  Fact 2.1: During his brief public career, Joe Orton outraged audiences with his scandalous black comedies.
  Fact 2.2: During his brief public career, Joe Orton amused audiences with his scandalous black comedies.
Sentence 3:  The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
  Fact 3.0: The adjective 'Ortonesque' is sometimes used to refer to work characterised by a dark yet farcical cynicism.
  Fact 3.1: The cynicism is similarly dark yet farcical.
Sentence 4: Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary modernism, and is considered by many to be among the greatest novelists in the English language.
  Fact 4.0: Henry James, OM, was an American author.
  Fact 4.1: Henry James was born in 1843.
  Fact 4.2: Henry James died in 1916.
  Fact 4.3: Henry James was regarded as a key transitional figure between literary realism and literary modernism.
  Fact 4.4: Henry James is considered by many to be among the greatest novelists in the English language.
Sentence 5:  He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
  Fact 5.0: Henry James was the son of Henry James, Sr.
  Fact 5.1: Henry James was the brother of the renowned philosopher and psychologist William James.
  Fact 5.2: Henry James was the brother of diarist Alice James.

Retrieved items:
<item 1>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 1>

<item 2>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 2>

<item 3>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 3>

<item 4>
What the Butler Saw (play)
What the Butler Saw is a farce written by the English playwright Joe Orton.

 It was premièred at the Queen's Theatre in London on 5 March 1969.

 It was Orton's final play and the second to be performed after his death, following "Funeral Games" in 1968.
</item 4>

<item 5>
Edgar Allan Poe’s humorous “A Predicament,” originally “The Scythe of Time,” parodies Gothic sensation tales and is often paired with “How to Write a Blackwood
Article.” British writer Michael Moorcock (born 1939), famed for Elric, wrote the science-fiction novels *Behold the Man* (expanded from a 1966 novella; title
from John’s Gospel), *The Final Programme*, *The Warlord of the Air* (1971; an
</item 5>

<item 6>
Prick Up Your Ears (play)
Prick Up Your Ears is a play by Simon Bent, based on the life of playwright Joe Orton.

 Produced by Sonia Friedman it opened at the Comedy Theatre in London's West End on 30 September 2009 following previews from 17 September.

 It starred Chris New as Joe Orton and Matt Lucas as Orton's lover and murderer, Kenneth Halliwell.
</item 6>

<item 7>
Prick Up Your Ears
Prick Up Your Ears is a 1987 British film, directed by Stephen Frears, about the playwright Joe Orton and his lover Kenneth Halliwell.

 The screenplay was written by Alan Bennett, based on the biography by John Lahr.

 The film stars Gary Oldman as Orton, Alfred Molina as Halliwell, Wallace Shawn as Lahr and Vanessa Redgrave as Peggy Ramsay.
</item 7>

<item 8>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 8>

<item 9>
The Ruffian on the Stair
The Ruffian On the Stair is a play by British playwright Joe Orton which was first broadcast on BBC Radio in August 1964.

 It is an unsympathetic yet comedic one-act portrayal of working class England, as played out by a couple and a mysterious young man who toys with their lives.

 It was based on "The Boy Hairdresser", a novel by Orton and his lover Kenneth Halliwell.

 The title and play are based on a few lines from poet and dramatist William Ernest Henley: "Madam Life's a piece in bloom, / Death goes dogging everywhere: /
She's the tenant of the room, / He's the ruffian on the stair."

 "Ruffian" is not as renowned as other works such as "Loot" and "What the Butler Saw", but it is still staged on occasion.
</item 9>

<item 10>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 10>
=== END ITEM id=e1e5dc593d74da46 ===

=== ITEM id=be0f9da556b0e1d4 ===
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
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 1>

<item 2>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
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
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 4>

<item 5>
Still Kicking: The Fabulous Palm Springs Follies
Still Kicking: The Fabulous Palm Springs Follies is a 1997 American short documentary film directed by Mel Damski.

 It was nominated at the 70th Academy Awards for Best Documentary Short Subject.

 It features The Fabulous Palm Springs Follies, which was formerly staged at the Palm Springs, California Plaza Theatre.
</item 5>

<item 6>
Time Bomb High School
Time Bomb High School is the second album by the Reigning Sound.

 It was released in 2002 on In the Red Records.

 The album featured the original Reigning Sound lineup of Greg Cartwright on lead vocals and guitar; Alex Greene on organ, piano, guitar, and backing vocals;
Jeremy Scott on bass, and backing vocals; and Greg Roberson on drums.

  Howlin' Pelle Almqvist of The Hives commented in "Rolling Stone" that Time Bomb High School was his favorite record of 2002.
</item 6>

<item 7>
Graph query row:
film: Still Kicking: Six Artistic Women of Project Arts & Longevity
release_year: 2006
connection_relationships: None
connection_path: None
connected_film: Time Bombs
connected_release_year: 2008
</item 7>

<item 8>
The Time to Live and the Time to Die
The Time to Live and the Time to Die, also known as A Time to Live, A Time to Die is a 1985 film directed by Hou Hsiao-hsien.

 This film is inspired by screenwriter-turned-director Hou's own coming-of-age story.
</item 8>

<item 9>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 9>

<item 10>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 10>
=== END ITEM id=be0f9da556b0e1d4 ===

=== ITEM id=f40e3ab399603a5a ===
Gold sentences and facts:
Sentence 0: Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with punctate stigmas on the female flowers.
  Fact 0.0: Lithocarpus is a genus in the beech family Fagaceae.
  Fact 0.1: Lithocarpus differs from "Quercus" in the erect spikes of insect-pollinated male flowers.
  Fact 0.2: Lithocarpus differs from "Quercus" in the short styles with punctate stigmas on the female flowers.
Sentence 1:  It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.
  Fact 1.0: Lithocarpus contains 17 species of shrubs and small trees.
  Fact 1.1: The Lithocarpus species are native from southern Florida to Mexico and South America.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Lithocarpus crassinervius
Lithocarpus crassinervius is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 2>

<item 3>
Lithocarpus glaber
Lithocarpus glaber, the Japanese oak, is a tree species in the genus "Lithocarpus" found in Japan, China and Taïwan.
</item 3>

<item 4>
Lithocarpus platycarpus
Lithocarpus platycarpus is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.

.
</item 4>

<item 5>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 5>

<item 6>
Lithocarpus kostermansii
Lithocarpus kostermansii is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 6>

<item 7>
List of poisonous plants
Plants cannot move to escape their predators, so they must have other means of protecting themselves from herbivorous animals.

 Some plants have physical defenses such as thorns, spines and prickles, but by far the most common type of protection is chemical.

 Over millennia, through the process of natural selection, plants have evolved the means to produce a vast and complicated array of chemical compounds in order
to deter herbivores.

 Tannin, for example, is a defensive compound that emerged relatively early in the evolutionary history of plants, while more complex molecules such as
polyacetylenes are found in younger groups of plants such as the Asterales.

 Many of the known plant defense compounds primarily defend against consumption by insects, though other animals, including humans, that consume such plants may
also experience negative effects, ranging from mild discomfort to death.
</item 7>

<item 8>
Duranta erecta
Duranta erecta is a species of flowering shrub in the verbena family Verbenaceae, native from Mexico to South America and the Caribbean.

 It is widely cultivated as an ornamental plant in tropical and subtropical gardens throughout the world, and has become naturalized in many places.

 It is considered an invasive species in Australia, China, South Africa and on several Pacific Islands.

 Common names include golden dewdrop, pigeon berry, and skyflower.

 In Mexico, the native Nahuatl name for the plant is xcambocoché.

 In Tonga it is known as mavaetangi (tears of departure).
</item 8>

<item 9>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 9>

<item 10>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 10>
=== END ITEM id=f40e3ab399603a5a ===

=== ITEM id=f313c374649d723c ===
Gold sentences and facts:
Sentence 0: Sir Ridley Scott (born 30 November 1937) is an English film director and producer.
  Fact 0.0: Sir Ridley Scott was born on 30 November 1937.
  Fact 0.1: Sir Ridley Scott is English.
  Fact 0.2: Sir Ridley Scott is a film director.
  Fact 0.3: Sir Ridley Scott is a film producer.
Sentence 1:  Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
  Fact 1.0: Ridley Scott had a commercial breakthrough with the science-fiction horror film Alien, released in 1979.
  Fact 1.1: Ridley Scott's best known works include the neo-noir dystopian science fiction film Blade Runner, released in 1982.
  Fact 1.2: Ridley Scott's best known works include the historical drama Gladiator, released in 2000.
  Fact 1.3: Gladiator won the Best Picture Oscar.
  Fact 1.4: Ridley Scott's best known works include the science fiction film The Martian, released in 2015.
Sentence 2:  A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work behind the camera, with Griffith and Joseph Henabery as his mentors.
  Fact 2.0: Ridley Scott was a collaborator of D.W. Griffith.
  Fact 2.1: Ridley Scott appeared in The Birth of a Nation, released in 1915.
  Fact 2.2: Ridley Scott appeared in Intolerance, released in 1916.
  Fact 2.3: Ridley Scott gave up acting in 1917.
  Fact 2.4: Ridley Scott gave up acting to concentrate on work behind the camera.
  Fact 2.5: Griffith was a mentor of Ridley Scott.
  Fact 2.6: Joseph Henabery was a mentor of Ridley Scott.

Retrieved items:
<item 1>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 1>

<item 2>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 2>

<item 3>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 3>

<item 4>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 4>

<item 5>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 5>

<item 6>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 6>

<item 7>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 7>

<item 8>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 8>

<item 9>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 9>

<item 10>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 10>
=== END ITEM id=f313c374649d723c ===

=== ITEM id=63f048bd030cdbe8 ===
Gold sentences and facts:
Sentence 0: Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to 2003.
  Fact 0.0: Václav Havel was born on 5 October 1936.
  Fact 0.1: Václav Havel died on 18 December 2011.
  Fact 0.2: Václav Havel was a Czech writer.
  Fact 0.3: Václav Havel was a former dissident.
  Fact 0.4: Václav Havel served as the 1st President of the Czech Republic.
  Fact 0.5: Václav Havel served as President of the Czech Republic from 1993 to 2003.
Sentence 1: Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.
  Fact 1.0: Penelope Fitzgerald was born on 17 December 1916.
  Fact 1.1: Penelope Fitzgerald died on 28 April 2000.
  Fact 1.2: Penelope Fitzgerald was English.
  Fact 1.3: Penelope Fitzgerald was a Booker Prize-winning novelist.
  Fact 1.4: Penelope Fitzgerald was a poet.
  Fact 1.5: Penelope Fitzgerald was an essayist.
  Fact 1.6: Penelope Fitzgerald was a biographer.

Retrieved items:
<item 1>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 1>

<item 2>
Letters to Olga
Letters to Olga ("Czech:Dopisy Olze") is a book of compiled letters written by Czech playwright, dissident, and future president, Václav Havel to his wife Olga
Havlová during his nearly four-year imprisonment from May 1979 to March 1983.

 (Havel was released when he came down with a high fever and received a medical discharge).

 Havel was imprisoned by the communist regime of then Czechoslovakia for being one of the leaders of The Committee for the Defense of the Unjustly Prosecuted
(VONS) - most of whom had been signatories of the human rights document Charter 77.
</item 2>

<item 3>
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
</item 3>

<item 4>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 4>

<item 5>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 5>

<item 6>
Václav Havel Tribute Concert
Tribute to Václav Havel (Czech: Pocta Václavu Havlovi) was an event held in memory of Václav Havel, the last Czechoslovak and the first Czech President, writer,
playwright and human rights activist.

 The concert took place in Lucerna Music Bar in Prague on 23 December 2011.

 Václav Havel died on Sunday 18 December 2011 at 10:15 a.m.
</item 6>

<item 7>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 7>

<item 8>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 8>

<item 9>
Václav Havel Human Rights Prize
The Václav Havel Human Rights Prize is an annual €60,000 award established in 2013 by the Parliamentary Assembly of the Council of Europe, the Václav Havel
Library and the Charta 77 Foundation, to honour "outstanding" civil society action in defence of human rights, in Europe and beyond.
</item 9>

<item 10>
Tereza Vilišová
Tereza Vilišová (born 1981) is a Czech actress.

 She won the Alfréd Radok Award for Best Actress in 2013 for her role of "Amy" in the D C Jackson play "My Romantic History" at the Divadlo Petra Bezruče in
Ostrava.

 At the 2013 Thalia Awards she won the category of Best Actress in a Play for the same work.

 Vilišová joined Prague's National Theatre in 2015.
</item 10>
=== END ITEM id=63f048bd030cdbe8 ===

=== ITEM id=cb3ba76e08433a79 ===
Gold sentences and facts:
Sentence 0: The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
  Fact 0.0: The Nashville Tribute Band is a Mormon music group.
  Fact 0.1: The Nashville Tribute Band was founded by Jason Deere.
  Fact 0.2: The Nashville Tribute Band was founded by Dan Truman.
  Fact 0.3: Dan Truman is the pianist of the popular country group Diamond Rio.
Sentence 1: Diamond Rio is an American country and Christian country music band.
  Fact 1.0: Diamond Rio is an American country music band.
  Fact 1.1: Diamond Rio is a Christian country music band.
Sentence 2:  The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then the Tennessee River Boys.
  Fact 2.0: Diamond Rio was founded in 1982.
  Fact 2.1: Diamond Rio was founded as an attraction for the Opryland USA theme park in Nashville, Tennessee.
  Fact 2.2: Diamond Rio was originally known as the Grizzly River Boys.
  Fact 2.3: Diamond Rio was later known as the Tennessee River Boys.

Retrieved items:
<item 1>
The Boys Are Back (The Oak Ridge Boys album)
The Boys Are Back is the thirtieth studio album of country music group The Oak Ridge Boys.

 It was released in 2009 under the Spring Hill Music Group label.

 The album marked the group's return to secular country music after releasing gospel albums since 1992.

 The track "Seven Nation Army," a cover of The White Stripes's 2003 song from their album "Elephant", was released as the album's first single.
</item 1>

<item 2>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 2>

<item 3>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 3>

<item 4>
Nashville Tribute Band
The Nashville Tribute Band is a Mormon music group founded by Jason Deere and Dan Truman, the pianist of the popular country group Diamond Rio.
</item 4>

<item 5>
Brother Clyde (album)
Brother Clyde is the debut studio album from alternative rock group Brother Clyde.

 It is the first side-project from country music singer Billy Ray Cyrus.

 The album features duets with Cyrus' son Trace Cyrus as well as Dolly Parton, who Cyrus said "wanted to rock."

 It was released on August 10, 2010.

 Its first single, "Lately", which features rapper King Phaze, was issued to digital retailers on June 29, 2010, and a music video, which was directed by Cyrus,
was released for the song in July 2010.
</item 5>

<item 6>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 6>

<item 7>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 7>

<item 8>
The Pilgrims (band)
The Pilgrims are an alternative rock band from Windsor, Vermont.

 Seven Days contributor Dan Bolles compared their style to alt-punk and indie rock, while John Powell, writing for www.angelica-music.com, has likened them to
rock and roll bands from the 1960s and 1970s.

 They are part of the Windsor, VT music collective What Doth Life, which has released albums by The Pilgrims, Giant Travel Avant Garde, Derek and The Demons,
Carton, and Luke Chrisinger.
</item 8>

<item 9>
The Studdogs
The Studdogs was a rock band from Orlando, Florida.

 The band's music contains elements of punk, blues rock, garage rock, and noise rock.

 The band was formed in 2000 and stayed active until 2005.

 They released a lot of material, particularly the full-length album, "The Gospel According to the Studdogs" and the 7" vinyl 3-song EP, and gained local and
national notoriety.

 They were featured in several national music magazines, including Magnet and Amplifier.

 Comparisons were made to the Stooges, Dead Boys, and it was said they sounded "like absolute victory by way of slurring, Rolling Stones-y garage rock from
nowhere in particular."

 Their drunken, abrasive live shows often ended with someone bleeding or being thrown out of the venue.

 Their live shows were said to be "what Howling Wolf would sound like through a wall of noise and distortion."
</item 9>

<item 10>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 10>
=== END ITEM id=cb3ba76e08433a79 ===
