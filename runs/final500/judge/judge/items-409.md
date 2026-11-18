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

=== ITEM id=5024f6233bd21505 ===
Gold sentences and facts:
Sentence 0: The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.
  Fact 0.0: The Appenzeller Sennenhund is a medium-size breed of dog.
  Fact 0.1: The Appenzeller Sennenhund is one of the four regional breeds of Sennenhund-type dogs.
  Fact 0.2: The Sennenhund-type dogs come from the Swiss Alps.
Sentence 1:  The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
  Fact 1.0: The name Sennenhund refers to people called "Senn".
  Fact 1.1: The Senn people are herders in the Appenzell region of Switzerland.
Sentence 2: The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.
  Fact 2.0: The Drentsche Patrijshond is a versatile spaniel-type hunting dog.
  Fact 2.1: The Drentsche Patrijshond is from the Dutch province of Drenthe.

Retrieved items:
<item 1>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 1>

<item 2>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 2>

<item 3>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 3>

<item 4>
Schutzhund
Schutzhund (German for "protection dog") is a dog sport that was developed in Germany in the early 1900s as a breed suitability test for the German Shepherd
breed.

 The test would determine if the dog displayed the appropriate traits and characteristics of a proper working German Shepherd.

 Today, it is used as a sport where many breeds other than German Shepherd Dogs can compete, but it is such a demanding test that few dogs can pass.
</item 4>

<item 5>
Bernese Mountain Dog
The Bernese Mountain Dog (German: "Berner Sennenhund" ) is a large-sized breed of dog, one of the four breeds of Sennenhund-type dogs from the Swiss Alps.

 The name "Sennenhund" is derived from the German "Senne" ("alpine pasture") and "Hund" ("dog"), as they accompanied the alpine herders and dairymen called
"Senn".

 "Berner" (or "Bernese" in English) refers to the area of the breed’s origin, in the canton of Bern.

 This mountain dog was originally kept as a general farm dog.

 Large Sennenhunde in the past were also used as draft animals, pulling carts.

 The breed was officially established in 1907.

 In 1937, the American Kennel Club recognized it; today, the club classifies it as a member of the Working Group.
</item 5>

<item 6>
Appenzeller (chicken)
The Appenzeller is a breed of chicken originating in Appenzell region of Switzerland.

 The Appenzeller comes in two varieties.

 The Spitzhauben variety, meaning "pointed hood", has a V-comb and feather crests in males and females.

 The word 'spitzhauben' derives from the frilly hat worn by the women in the Appenzeller region in Switzerland.

 The breed was imported into America by a doctor.

 The Barthuhner ("bearded hen") has a rose comb and no crest.

 Both types appear in either black, golden spangled and silver spangled plumage.

 They are mostly a show breed, but are decent egg layers.

 They lay small white eggs about 5 times per week.
</item 6>

<item 7>
Entlebucher Mountain Dog
The Entlebucher Sennenhund or Entlebucher Mountain Dog is a medium-sized herding dog, it is the smallest of the four Sennenhunds, a dog type that includes four
regional breeds.

 The name Sennenhund refers to people called "Senn", herders in the Swiss Alps.

 Entlebuch is a region in the canton of Lucerne in Switzerland.

 The breed is also known in English as the Entelbuch Mountain Dog, Entelbucher Cattle Dog, and similar combinations.
</item 7>

<item 8>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 8>

<item 9>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 9>

<item 10>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 10>
=== END ITEM id=5024f6233bd21505 ===

=== ITEM id=c7bb1bd0b14a8ff5 ===
Gold sentences and facts:
Sentence 0: John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).
  Fact 0.0: John H. Garabedian is an American radio personality.
  Fact 0.1: John H. Garabedian is an American disc jockey.
  Fact 0.2: John H. Garabedian was born on December 20, 1941.
  Fact 0.3: John H. Garabedian is age 75.
Sentence 1: Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
  Fact 1.0: Charles A. Agemian was born July 22, 1909.
  Fact 1.1: Charles A. Agemian's life ended March 30, 1996.
  Fact 1.2: Charles A. Agemian was an American banker.
  Fact 1.3: Charles A. Agemian took early retirement from Chase Manhattan Bank.
  Fact 1.4: Charles A. Agemian was executive vice president of operations at Chase Manhattan Bank.
  Fact 1.5: Charles A. Agemian became chairman of the Hackensack Trust Co.
  Fact 1.6: Charles A. Agemian became chief executive officer of the Hackensack Trust Co.
  Fact 1.7: The Hackensack Trust Co. was later renamed Garden State National Bank.

Retrieved items:
<item 1>
Graph query row:
person: John H. Garabedian
profession: ['radio personality', 'disc jockey']
relationship: CREATED_BY
role: None
related_entity: Open House Party
</item 1>

<item 2>
Don Sherwood (DJ)
Don Sherwood (September 7, 1925 – November 6, 1983) was an American radio personality.

 He was a San Francisco, California, disc jockey during the 1950s and 1960s.

 Billed as "The World's Greatest Disc Jockey," Sherwood spent most of his career hosting a 6-9 a.m. weekday program on KSFO in San Francisco (560 kHz, 5000
watts), which was then owned by the singing cowboy actor Gene Autry.
</item 2>

<item 3>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 3>

<item 4>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 4>

<item 5>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 5>

<item 6>
Radio personality
A radio personality (American English) or radio presenter (British English), commonly referred to as a "disc jockey" or "DJ" for short, is a person who has an
on-air position in radio broadcasting.

 A radio personality that hosts a radio show is also known as a radio host, and in India and Pakistan as a radio jockey.

 Radio personalities who introduce and play individual selections of recorded music are known as disc jockeys.

 The term has evolved to also describe a person who mixes a continuous flow of recorded music in real time.

 Broadcast radio personalities may include talk radio hosts, AM/FM radio show hosts, and satellite radio program hosts.

 Notable radio personalities include pop music radio hosts Martin Block, Alan Freed, Dick Clark, Delilah Luke, Ameen Sayani, Wolfman Jack, and Casey Kasem,
shock jocks such as Don Imus and Howard Stern, as well as sports talk hosts such as Mike Francesa and political talk hosts such as Rush Limbaugh.
</item 6>

<item 7>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 7>

<item 8>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 8>

<item 9>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 9>

<item 10>
M. G. Kelly
Gary Sinclair (born 1952 in Ada, Oklahoma), known professionally as Michael Gary “M.G.” Kelly and Machine Gun Kelly, is an American actor, disc jockey and radio
personality.

 In addition to hosting several radio programs over the years, Kelly has held several acting roles as a disc jockey; also, he has served as an offstage
announcer on two game shows.
</item 10>
=== END ITEM id=c7bb1bd0b14a8ff5 ===

=== ITEM id=803189d8408e80be ===
Gold sentences and facts:
Sentence 0: Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.
  Fact 0.0: Semyon Aronovich Gershgorin was born on August 24, 1901.
  Fact 0.1: Semyon Aronovich Gershgorin died on May 30, 1933.
  Fact 0.2: Semyon Aronovich Gershgorin was Soviet.
  Fact 0.3: Semyon Aronovich Gershgorin was born in Pruzhany, Belarus, Russian Empire.
  Fact 0.4: Semyon Aronovich Gershgorin was a mathematician.
Sentence 1:  He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after him.
  Fact 1.0: Semyon Aronovich Gershgorin was a mathematician.
  Fact 1.1: Semyon Aronovich Gershgorin was noted for his work on the decidability of various algebraic groups.
  Fact 1.2: Malcev algebras are generalisations of Lie algebras.
  Fact 1.3: Malcev algebras are named after Semyon Aronovich Gershgorin.

Retrieved items:
<item 1>
Anna Aglatova
Anna Khachaturovna Aglatova (Russian: Анна Хачатуровна Аглатова ) is a Russian soprano singer who was born in Kislovodsk and by 2004 joined the singing
department of the Gnessin State Musical College.

 Prior to it, she was a recipient of the "Sergei Leiferkus grant" from the Vladimir Spivakov Fund and in 2005 made her first public appearance at the Bolshoi
Theatre.

 Her career there, have not started with ease since her father didn't want her to work for Bolshoi.

 In 2003, she was a recipient of the first prize at the Bella voce International Competition and the same year participated at both the Christmas Festival at
Düsseldorf and the fourteenth annual Chaliapin Season at Mineralnye Vody.

 She also was a participant for "Irina Arkhipova Fund" and was a part of the Novosibirsk Opera and Ballet Theatre.
</item 1>

<item 2>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 2>

<item 3>
Alexander Artemev
Alexander Vladimirovich "Sasha" Artemev (Russian: Александр Владимирович "Саша" Артемьев , Belarusian: Аляксандр Уладзіміравіч Арцем'еў ; born August 29, 1985)
is a retired American artistic gymnast.

 Artemev was a member of the bronze medal-winning U.S. team at the 2008 Olympic Games.

 He is the 2006 all-around U.S. national champion.

 Known for his ability on pommel horse, he is the 2007 and 2008 U.S. national champion on the pommel horse and won the bronze medal on the event at the 2006
World Championships.
</item 3>

<item 4>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 4>

<item 5>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 7>

<item 8>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 8>

<item 9>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 9>

<item 10>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 10>
=== END ITEM id=803189d8408e80be ===

=== ITEM id=c98c50ab4646ec81 ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Cage starred in a variety of films.
  Fact 0.1: Cage starred in Valley Girl, released in 1983.
  Fact 0.2: Cage starred in Racing with the Moon, released in 1984.
  Fact 0.3: Cage starred in Birdy, released in 1984.
  Fact 0.4: Cage starred in Peggy Sue Got Married, released in 1986.
  Fact 0.5: Cage starred in Raising Arizona, released in 1987.
  Fact 0.6: Cage starred in Moonstruck, released in 1987.
  Fact 0.7: Cage starred in Vampire's Kiss, released in 1989.
  Fact 0.8: Cage starred in Wild at Heart, released in 1990.
  Fact 0.9: Cage starred in Fire Birds, released in 1990.
  Fact 0.10: Cage starred in Honeymoon in Vegas, released in 1992.
  Fact 0.11: Cage starred in Red Rock West, released in 1993.
Sentence 1: Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.
  Fact 1.0: Moonstruck is a 1987 American film.
  Fact 1.1: Moonstruck is a romantic comedy film.
  Fact 1.2: Moonstruck was directed by Norman Jewison.
  Fact 1.3: Moonstruck was written by John Patrick Shanley.
Sentence 2:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 2.0: Moonstruck is about a widowed 37-year-old Italian-American woman played by Cher.
  Fact 2.1: The woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 2.2: The fiancé is played by Danny Aiello.
  Fact 2.3: The younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 1>

<item 2>
Nagaram (2010 film)
Nagaram Marupakkam (English: The other side of the town ) is a 2010 Indian Tamil action film co-produced written and directed by Sundar C., who himself plays
the lead role.

 Co-produced by his wife Kushboo, the film also stars Anuya Bhagvath and Vadivelu in pivotal roles and features music by Thaman.

 Sundar's first directorial after five years, "Nagaram" released on 19 November 2010, it is based on the 1993 American movie Carlito's Way .

 The film is very well known for comedy sequences and portrayal of the comic character "Style Pandi" by Vadivelu.
</item 2>

<item 3>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 3>

<item 4>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 4>

<item 5>
Dying of the Light (film)
Dying of the Light is a 2014 American psychological thriller film written and directed by Paul Schrader and starring Nicolas Cage, Anton Yelchin and Irène Jacob
about a government agent who must track down and kill a terrorist before he loses his full memory from a disease.

 It was released theatrically and through VOD formats by Lionsgate on December 5, 2014.

 The film received extremely negative reviews, with controversy surrounding the heavy tampering and reediting of the footage by the studio, who denied Schrader
final-cut privilege and led him and principal members of the cast to disown the released version and campaign against it.
</item 5>

<item 6>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 6>

<item 7>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 7>

<item 8>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 8>

<item 9>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 9>

<item 10>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 10>
=== END ITEM id=c98c50ab4646ec81 ===

=== ITEM id=7e536697e023049a ===
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
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 1>

<item 2>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 2>

<item 3>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 3>

<item 4>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 4>

<item 5>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 5>

<item 6>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 6>

<item 7>
Chiguata District
Cherubs also decorate the circular cornice, along with four saints in full-length reliefs that occupy the "pechinas" of the columns that sustain the temple's
dome, which is considered one of the most original works of the colony and now visited by occasional tourists.
</item 7>

<item 8>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 8>

<item 9>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 9>

<item 10>
Chiguata District
Chiguata is a district located in the Arequipa province, 30 km away from the city of Arequipa, Peru, on the slopes of the volcano Pikchu Pikchu (where at its
summit pre-Inca worshipped remains and artifacts have been found).

 The name "chiguata" comes from the Quechua words "chiri" meaning "cold" and "wata" meaning "year" (cold all year).

 Ancient Chiguata was inhabited by the local natives until the arrival of Spanish influence in charge of the emissary Diego Hernández de Córdova in 1540.

 Soon after, the construction of a small plaza was initiated, and according to the Dominican religious order and local sources, the plaza's "Templo del Espíritu
Santo" ("Sacred Spirit temple") along with its domes and lateral cover was not begun until 1739.

 The design of the temple is quite interesting, with two angels in feathered nightwear and their arms lifted vertically, separated by twelve fringes of
rectangular flowers which add to the whole decoration of the temple.
</item 10>
=== END ITEM id=7e536697e023049a ===

=== ITEM id=2014ed88785756f1 ===
Gold sentences and facts:
Sentence 0: Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban parents in France, where she was also raised.
  Fact 0.0: Anaïs Nin was born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell.
  Fact 0.1: Anaïs Nin was born on February 21, 1903.
  Fact 0.2: Anaïs Nin died on January 14, 1977.
  Fact 0.3: Anaïs Nin was an essayist.
  Fact 0.4: Anaïs Nin was a memoirist.
  Fact 0.5: Anaïs Nin was born to Cuban parents.
  Fact 0.6: Anaïs Nin was born in France.
  Fact 0.7: Anaïs Nin was raised in France.
Sentence 1: Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as a major exponent of romantic modernism in Greece and the world.
  Fact 1.0: Odysseas Elytis is a pen name of Odysseas Alepoudellis.
  Fact 1.1: Odysseas Elytis's Greek name is Οδυσσέας Ελύτης.
  Fact 1.2: Odysseas Alepoudellis's Greek name is Οδυσσέας Αλεπουδέλλης.
  Fact 1.3: Odysseas Elytis was born on 2 November 1911.
  Fact 1.4: Odysseas Elytis died on 18 March 1996.
  Fact 1.5: Odysseas Elytis was regarded as a major exponent of romantic modernism in Greece.
  Fact 1.6: Odysseas Elytis was regarded as a major exponent of romantic modernism in the world.
Sentence 2:  In 1979 he was awarded the Nobel Prize in Literature.
  Fact 2.0: Odysseas Elytis was awarded the Nobel Prize in Literature in 1979.

Retrieved items:
<item 1>
Seduction of the Minotaur
Seduction of the Minotaur is an autobiographical novel by the mixed nationality writer Anaïs Nin, the last part of her "Cities of the Interior" sequence.

 It is about a woman named Lillian, and her self-psychoanalysis.

 The setting is taken from Anaïs' diary account of her first trip to Acapulco in 1947, and the novel repeats much of the first part of "The Diary of Anaïs Nin"
volume V.

 Since the author was concerned with psychology rather than physical adventure, there is actually less violence in the novel than in the diary account.

 The exception is that the doctor allows himself to be shot because he is loved only as a doctor and never as a man, perhaps patterned after her understanding
of Otto Rank's death.
</item 1>

<item 2>
Nearer the Moon
Nearer the Moon: From a Journal of Love (full title "Nearer the Moon: From a Journal of Love, the Unexpurgated Diary of Anaïs Nin (1937–1939)") is a 1996 book
based upon material excerpted from the unpublished diaries of Anaïs Nin.

 It corresponds temporally to part of Nin's published diaries.

 It consists mainly of material that was left out of the published version because it would have hurt people involved or their relationships with Anaïs Nin had
it been published at the time.
</item 2>

<item 3>
The Diary of Anaïs Nin
The Diary of Anaïs Nin is the published version of Anaïs Nin's own private manuscript diary, which she began at age 11 in 1914 during a trip from Europe to New
York with her mother and two brothers.

 Anaïs Nin would later say she had begun the diary as a letter to her father, Cuban composer Joaquín Nin, who had abandoned the family a few years earlier.
</item 3>

<item 4>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 4>

<item 5>
Odysseas Elytis
Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as
a major exponent of romantic modernism in Greece and the world.

 In 1979 he was awarded the Nobel Prize in Literature.
</item 5>

<item 6>
Henry and June
Henry and June: From the Unexpurgated Diary of Anaïs Nin (full title "Henry and June: From A Journal of Love: the Unexpurgated Diary of Anaïs Nin (1931–1932)")
is a 1986 book that is based upon material excerpted from the unpublished diaries of Anaïs Nin.
</item 6>

<item 7>
Anaïs Nin: A Biography
Anaïs Nin: A Biography is Deirdre Bair's award-winning biography of writer Anaïs Nin.

 It is considered arguably by many to be the most comprehensive, well-researched, and scholarly biography available of Nin.

 Though the biography has received praise, it has also angered some fans of Anaïs Nin as well as some of her former associates, some of whom claim that Bair's
critical and rigorous investigation of Anaïs Nin's life is unsympathetic.
</item 7>

<item 8>
Collages (novel)
Collages, published in 1964, was Anaïs Nin's last published novel (excluding her erotica).

 It is very different from the previous novels of the "Cities of the Interior" series, because it contains none of the familiar characters in those novels.

 It is also different in that it takes place on two continents, has about two dozen important characters, and for the first time in Anaïs Nin's work the female
lead is not seeking psychological wholeness.

 Most of Anaïs Nin's other novels only have a few main characters, there is rarely much geographical movement in the other novels, and usually the lead
character is a woman seeking psychological wholeness.
</item 8>

<item 9>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 9>

<item 10>
Incest: From a Journal of Love
Incest: From a Journal of Love: The Unexpurgated Diary of Anaïs Nin (1932–1934) is a 1992 non-fiction book by Anaïs Nin.

 It is a continuation of the diary entries first published in "Henry and June: From the Unexpurgated Diary of Anaïs Nin".

 It features Nin's relationships with writer Henry Miller, his wife June Miller, the psychoanalyst Otto Rank, her father Joaquín Nin, and her husband Hugh
Parker Guiler.

 She also copied some of her correspondence with these people into her diary.

 Much of this book was written in English, although those of her letters which were originally written in French and Spanish were translated.

 Most of this diary takes place in France, particularly Clichy, Paris and Louveciennes.
</item 10>
=== END ITEM id=2014ed88785756f1 ===

=== ITEM id=fe84e7ad3c077e1a ===
Gold sentences and facts:
Sentence 0:  The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and Viv Albertine, who went on to play in The Slits.
  Fact 0.0: The band never played live.
  Fact 0.1: The band never released any recordings.
  Fact 0.2: Like London SS, the band is more famed for the number of band members who later became well known.
  Fact 0.3: Like Masters of the Backside, the band is more famed for the number of band members who later became well known.
  Fact 0.4: Sid Vicious is listed as a band member who later became well known as a member of the Sex Pistols.
  Fact 0.5: Keith Levene is listed as a band member who later became well known.
  Fact 0.6: Keith Levene was an early member of The Clash.
  Fact 0.7: Keith Levene was later a member of Public Image Ltd.
  Fact 0.8: Palmolive is listed as a band member who later became well known.
  Fact 0.9: Viv Albertine is listed as a band member who later became well known.
  Fact 0.10: Viv Albertine went on to play in The Slits.
Sentence 1:  Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
  Fact 1.0: The band was interviewed by a fanzine named SKUM despite never playing live.
  Fact 1.1: In the SKUM interview, Sid Vicious proclaimed 'I'll just be the yob that I am now'.
Sentence 2: Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock band the Sex Pistols.
  Fact 2.0: Sid Vicious was born John Simon Ritchie.
  Fact 2.1: Sid Vicious was born on 10 May 1957.
  Fact 2.2: Sid Vicious died on 2 February 1979.
  Fact 2.3: Sid Vicious was an English bassist.
  Fact 2.4: Sid Vicious was an English vocalist.
  Fact 2.5: Sid Vicious was most famous as a member of the influential punk rock band the Sex Pistols.

Retrieved items:
<item 1>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 1>

<item 2>
The Swindle Continues
The Swindle Continues is the first album released by the Ex Pistols in 1988.

 It consists of half Sex Pistols and half Ex Pistols material, and is the only Ex Pistols release that doesn't disguise itself as material completely by the Sex
Pistols.
</item 2>

<item 3>
Sid Vicious
Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock
band the Sex Pistols.
</item 3>

<item 4>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 4>

<item 5>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 5>

<item 6>
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 6>

<item 7>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 7>

<item 8>
Revolution in the Classroom
"Revolution in the Classroom" was a single released by the Ex Pistols; despite its packaging, the record has nothing to do with the Sex Pistols.

 The single was released in 1989 on the STP Records label, pressed on various colours of vinyl.

 The song, along with its B-side "Schools Are Prisons", were both included on the Ex-Pistols album, "Deny".

 They also appear disguised as genuine Sex Pistols recordings on two Sex Pistols retrospective albums by former producer Dave Goodman, "Pirates of Destiny" (I
Swirled Records, 1989) and "We Have Cum For Your Children" (Skyclad Records, 1988).

Under the title "Revolution" the song appears on a German Sex Pistols compilation "Anarchy in the USA" from 1992 released on MBC records.

 CD 084-60322 LP 008-60321
</item 8>

<item 9>
Bodies (Sex Pistols song)
"Bodies" is a Sex Pistols song about the shock of abortion from the 1977 album "Never Mind the Bollocks, Here's the Sex Pistols".

 The lyrics contain very graphic imagery about a terminated fetus and feature a great deal of profanity for the time: the third and final verse begins with a
couplet in which the word "fuck" is repeated five times in rapid succession.

 Along with the later "Belsen Was a Gas," it is probably the most graphic and controversial Sex Pistols song in both its subject matter and style.

 Musically, it is also the fastest and heaviest song in the Sex Pistols canon — characterized by thudding drums, droning buzzsaw guitar, and shouted vocals.

 As such, it can be considered a significant antecedent to the genres of hardcore, thrash, and speed metal that was to emerge in the 1980s.
</item 9>

<item 10>
Ex Pistols
The Ex Pistols were a scam act introduced in 1979 by one-time Sex Pistols producer Dave Goodman after his services were substituted for those of professional
producers.

 The Ex Pistols existed as a sound-alike band meant to mislead fans due to the similar music, name, and artwork, that led people to believe their songs were
actually new or lost Sex Pistols recordings.
</item 10>
=== END ITEM id=fe84e7ad3c077e1a ===

=== ITEM id=050421108c4a120e ===
Gold sentences and facts:
Sentence 0: Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.
  Fact 0.0: Ravi Sethi was born in 1947.
  Fact 0.1: Ravi Sethi is an Indian computer scientist.
  Fact 0.2: Ravi Sethi retired from Bell Labs.
  Fact 0.3: Ravi Sethi is president of Avaya Labs Research.
Sentence 1: Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company, owned by Finnish company Nokia.
  Fact 1.0: Nokia Bell Labs was formerly named AT&T Bell Laboratories.
  Fact 1.1: Nokia Bell Labs was formerly named Bell Telephone Laboratories.
  Fact 1.2: Nokia Bell Labs was formerly named Bell Labs.
  Fact 1.3: Nokia Bell Labs is an American research and scientific development company.
  Fact 1.4: Nokia Bell Labs is owned by the Finnish company Nokia.
Sentence 2:  Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
  Fact 2.0: The headquarters of Nokia Bell Labs are located in Murray Hill, New Jersey.
  Fact 2.1: Nokia Bell Labs has other laboratories around the rest of the United States.
  Fact 2.2: Nokia Bell Labs has laboratories in other countries.

Retrieved items:
<item 1>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 1>

<item 2>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 2>

<item 3>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 3>

<item 4>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 4>

<item 5>
David H. Bailey
David Harold Bailey (born 1948) is a mathematician and computer scientist.

 He received his B.S. in mathematics from Brigham Young University in 1972 and his Ph.D. in mathematics from Stanford University in 1976.

 He worked for 14 years as a computer scientist at NASA Ames Research Center, but since 1998 has been at the Lawrence Berkeley National Laboratory.

 He is now officially retired, but continues as an active researcher.

 He is also a Research Associate at the University of California, Davis, Department of Computer Science.
</item 5>

<item 6>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 6>

<item 7>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 7>

<item 8>
Ravi Sethi
Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.

 He also serves as a member of the National Science Foundation's Computer and Information Science and Engineering (CISE) Advisory Committee.

 He is best known as one of three authors of the classic computer science textbook "", also known as the "Dragon Book".
</item 8>

<item 9>
GeneDx
GeneDx is genetic testing company that was founded in 2000 by two scientists from the National Institutes of Health (NIH), Drs. Sherri Bale and John Compton.

 They specialize in testing for rare inherited disorders.

 In 2006, BioReference laboratories acquired GeneDx.

 Since then GeneDx has operated as a subsidiary of this parent company under the leadership of Drs. Bale and Compton.
</item 9>

<item 10>
Sethi–Ullman algorithm
In computer science, the Sethi–Ullman algorithm is an algorithm named after Ravi Sethi and Jeffrey D. Ullman, its inventors, for translating abstract syntax
trees into machine code that uses as few registers as possible.
</item 10>
=== END ITEM id=050421108c4a120e ===

=== ITEM id=1a0f6e9f56895ca3 ===
Gold sentences and facts:
Sentence 0:  In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site LiveJasmin.
  Fact 0.0: In the category of pornographic websites, this site was surpassed in the rankings by xHamster.
  Fact 0.1: In the category of pornographic websites, this site was surpassed in the rankings by XVideos.
  Fact 0.2: In the category of pornographic websites, this site was surpassed in the rankings by Pornhub.
  Fact 0.3: In the category of pornographic websites, this site was surpassed in the rankings by the adult webcam site LiveJasmin.
  Fact 0.4: xHamster, XVideos, and Pornhub are competitor sites.
Sentence 1: XVideos is the world's largest pornographic video sharing and viewing website.
  Fact 1.0: XVideos is the world's largest pornographic video sharing and viewing website.

Retrieved items:
<item 1>
YouPorn
YouPorn is a free pornographic video sharing website and one of the 100 most accessed websites in the world.

 Since launching in August 2006, it grew to become the most popular pornographic website on the internet, and, in November 2007, it was reported to be the
largest free pornographic website as well.

 As of February 2013, it was the 83rd most popular website overall and the fifth most popular pornographic website.

 In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site
LiveJasmin.
</item 1>

<item 2>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 2>

<item 3>
XY TV
It launched in December 2003 and distribution was limited to select cable companies.

 As of May 31, 2006, XY.tv has ceased operations, and is no longer broadcasting.

 It was wholly owned by American radio personality John Garabedian.
</item 3>

<item 4>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 4>

<item 5>
Pravda.ru
Pravda.ru (formerly Pravda Online) is a Russian internet news website established in 1999 and owned by Pravda.ru Holding (headed by Vadim Gorshenin).
</item 5>

<item 6>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 6>

<item 7>
Tempestt Bledsoe
Tempestt Bledsoe (born August 1, 1973) is an American actress.

 She is best known for her childhood role as Vanessa Huxtable, the fourth child of Cliff and Clair Huxtable on the long-running NBC sitcom "The Cosby Show"
(1984–92).

 In December 2010, it was announced that Bledsoe would be the host of "Clean House" on the Style Network, replacing long-time host Niecy Nash.

 From September 2012 to February 2013, she was one of the stars of the NBC TV sitcom "Guys with Kids", portraying Marny.
</item 7>

<item 8>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 8>

<item 9>
XHamster
xHamster is a pornographic media and social networking site headquartered in Limassol, Cyprus.

 xHamster serves user-submitted pornographic videos, webcam models, pornographic photographs, and erotic literature, and incorporates social networking
features.

 xHamster was founded in 2007.

 With more than 10 million members, it is the third most popular pornography website on the Internet after XVideos and Pornhub.
</item 9>

<item 10>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 10>
=== END ITEM id=1a0f6e9f56895ca3 ===

=== ITEM id=919591f40c3a8481 ===
Gold sentences and facts:
Sentence 0: Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.
  Fact 0.0: Broughtonia is a genus of orchids.
  Fact 0.1: Broughtonia belongs to the family Orchidaceae.
  Fact 0.2: Broughtonia is native to the Bahamas.
  Fact 0.3: Broughtonia is native to the Greater Antilles.
Sentence 1: Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family, Lamiaceae.
  Fact 1.0: Perilla is a genus.
  Fact 1.1: Perilla consists of one major Asiatic crop species, P. frutescens.
  Fact 1.2: Perilla includes a few recognized wild species in nature.
  Fact 1.3: Perilla belongs to the mint family, Lamiaceae.

Retrieved items:
<item 1>
Monardella lanceolata
Monardella lanceolata is a species of flowering plant in the mint family known by the common names mustang mint and mustang monardella.

 It is native to the mountains of California and Baja California, where it grows in chaparral, woodland, rocky slopes, and often disturbed habitat types.
</item 1>

<item 2>
Shiso
Perilla frutescens var.

 "crispa", also called shiso ( , from Japanese シソ ) is a variety of species "Perilla frutescens" of the genus "Perilla", belonging to the mint family,
Lamiaceae.

 The plant occurs in red (purple-leaved) or green-leaved forms.

 Shiso is a perennial plant that may be cultivated as an annual in temperate climates, and occurs in both red- (or purple-) leaved and green forms.

 There are also frilly, ruffled-leaved forms called "chirimen-jiso" and forms that are red only on top, called "katamen-jiso".
</item 2>

<item 3>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 3>

<item 4>
Lamiaceae
or Labiatae is a family of flowering plants commonly known as the mint or deadnettle family.

 Many of the plants are aromatic in all parts and include widely used culinary herbs, such as basil, mint, rosemary, sage, savory, marjoram, oregano, hyssop,
thyme, lavender, and perilla.

 Some species are shrubs, trees (such as teak), or, rarely, vines.

 Many members of the family are widely cultivated, not only for their aromatic qualities but also their ease of cultivation, since they are readily propagated
by stem cuttings.

 Besides those grown for their edible leaves, some are grown for decorative foliage, such as "Coleus".

 Others are grown for seed, such as "Salvia hispanica" (chia), or for their edible tubers, such as "Plectranthus edulis", "Plectranthus esculentus",
"Plectranthus rotundifolius", and "Stachys affinis" (Chinese artichoke).
</item 4>

<item 5>
Graph query row:
entity: Perilla
entity_types: ['__KGBuilder__', '__Entity__', 'Thing']
relationship: PART_OF
connected_to: Lamiaceae
connected_types: ['__KGBuilder__', '__Entity__', 'Thing']
</item 5>

<item 6>
Perilla
Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family,
Lamiaceae.

 It encompasses several distinct varieties of Asian herb, seed, and vegetable crop, including "P. frutescens" (deulkkae) and "P. frutescens" var.

 "crispa" (shiso).

 The genus name "Perilla" is also a frequently employed common name ("perilla"), applicable to all varieties.

 Perilla varieties are cross-fertile and intra-specific hybridization occurs naturally.

 Some varieties are considered invasive.
</item 6>

<item 7>
Broughtonia
Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.

 The genus is abbreviated Bro in trade journals.
</item 7>

<item 8>
Perilla ketone
Perilla ketone is a natural terpenoid that consists of a furan ring with a six-carbon side chain containing a ketone functional group.

 It is a colorless oil that is sensitive to oxygen, becoming colored upon standing.

 The ketone was identified in 1943 by Sebe as the main component of the essential oil of Perilla frutescens.

 Perilla ketone is present in the leaves and seeds of purple mint ("Perilla frutescens"), which is toxic to some animals.

 When cattle and horses consume purple mint when grazing in fields in which it grows, the perilla ketone causes pulmonary edema leading to a condition sometimes
called perilla mint toxicosis.
</item 8>

<item 9>
Perilla frutescens
Perilla frutescens, commonly called perilla or Korean perilla, is a species of "Perilla" in the mint family Lamiaceae.

 It is an annual plant native to Southeast Asia and Indian highlands, and grown in the Korean peninsula, Southern China and India.

 Known as "deulkkae" (들깨 ) in Korean, the plant was introduced into Korea before the Unified Silla era, when it started to be widely cultivated.
</item 9>

<item 10>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 10>
=== END ITEM id=919591f40c3a8481 ===

=== ITEM id=b959bf88afae915f ===
Gold sentences and facts:
Sentence 0: Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an SS-"Obergruppenführer".
  Fact 0.0: Karl Hermann Frank's life dates are 24 January 1898 to 22 May 1946.
  Fact 0.1: Karl Hermann Frank was a prominent Sudeten German Nazi official in Czechoslovakia.
  Fact 0.2: Karl Hermann Frank was a Nazi official in Czechoslovakia prior to World War II.
  Fact 0.3: Karl Hermann Frank was a Nazi official in Czechoslovakia during World War II.
  Fact 0.4: Karl Hermann Frank was an SS-Obergruppenführer.
Sentence 1:  Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".
  Fact 1.0: Until April 1942, the Obergruppenführer rank was the highest commissioned SS rank.
  Fact 1.1: Until April 1942, the Obergruppenführer rank was inferior only to Reichsführer-SS.
  Fact 1.2: Reichsführer-SS was the SS rank held by Heinrich Himmler.
  Fact 1.3: RFSS was the internal SS abbreviation for Himmler.
  Fact 1.4: Obergruppenführer translates as senior group leader.
  Fact 1.5: The rank of Obergruppenführer was senior to Gruppenführer.

Retrieved items:
<item 1>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 1>

<item 2>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 2>

<item 3>
Obergruppenführer
Obergruppenführer (] , "senior group leader") was a Nazi Party paramilitary rank that was first created in 1932 as a rank of the "Sturmabteilung" (SA), and
adopted by the "Schutzstaffel" (SS) one year later.

 Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation
for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".

 A similarly named rank of "Untergruppenführer" existed in the SA from 1929 to 1930 and as a title until 1933.

 In April 1942, the new rank of "SS-Oberst-Gruppenführer" was created which was above "Obergruppenführer" and below "Reichsführer-SS".
</item 3>

<item 4>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 4>

<item 5>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 5>

<item 6>
Frank A. Armstrong
Frank Alton Armstrong Jr. (May 24, 1902 – August 20, 1969) was a lieutenant general of the United States Air Force.

 As a brigadier general in the United States Army Air Forces during World War II, he was the inspiration for the main character in the novel and subsequent
film, "Twelve O'Clock High."

 After the war, he held a variety of senior leadership positions prior to and following the establishment of the USAF as an independent service in 1947.

 Promoted to major general in 1950, he advanced to lieutenant general in 1956 and retired at that rank in 1962.
</item 6>

<item 7>
Karl Hermann Frank
Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an
SS-"Obergruppenführer".

 He was tried, convicted and executed after World War II for his role in organizing the massacres of the people of the Czech villages of Lidice and Ležáky.
</item 7>

<item 8>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 8>

<item 9>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 9>

<item 10>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 10>
=== END ITEM id=b959bf88afae915f ===

=== ITEM id=f2b47a87b96e05be ===
Gold sentences and facts:
Sentence 0: Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.
  Fact 0.0: Batman is a 1989 American superhero film.
  Fact 0.1: Batman was directed by Tim Burton.
  Fact 0.2: Batman was produced by Jon Peters and Peter Guber.
  Fact 0.3: Batman is based on the DC Comics character of the same name.
Sentence 1:  He is Chairman and CEO of Mandalay Entertainment.
  Fact 1.0: The person referred to as 'he' is Chairman of Mandalay Entertainment.
  Fact 1.1: The person referred to as 'he' is CEO of Mandalay Entertainment.

Retrieved items:
<item 1>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 1>

<item 2>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 2>

<item 3>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 3>

<item 4>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 4>

<item 5>
Batman Forever (disambiguation)
Batman Forever is a 1995 film in the Batman film series.
</item 5>

<item 6>
Batman vs. Two-Face
Batman vs. Two-Face is an upcoming 2017 American animated direct-to-video superhero film produced by Warner Bros.

 Animation and distributed by Warner Bros.

 Home Entertainment.

 It is a sequel to "".

 It will premiere at the New York Comic Con on October 8, 2017 and will be followed by a digital release on October 10, 2017 and on DVD and Blu-ray on October
17, 2017.

 Based on the 1960s "Batman" TV series, the film stars Adam West (in his final role before his death), Burt Ward and Julie Newmar reprising their roles of
Batman, Robin and Catwoman from the series.

 The film will show a tribute to the late Adam West.

 This could also be the final time Warner Bros.

 Animation would make a 1960s Batman film.
</item 6>

<item 7>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 7>

<item 8>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 8>

<item 9>
Batman: Return of the Joker
Batman: Return of the Joker, known in Japan as Dynamite Batman (ダイナマイトバットマン , Dainamaito Battoman ) , is a 1991 platform video game, the follow-up to Sunsoft's
on the Nintendo Entertainment System.

 Unlike that game, which was based on the 1989 Batman film directed by Tim Burton, "Return of the Joker" is entirely self-contained and based more on the modern
comic book iteration of Batman.

 However, Batman mans the Batmobile from the 1989 film.

 A remake of "Return of the Joker", titled Batman: Revenge of the Joker, was released on the Sega Genesis by Ringler Studios in 1992.

 A Super NES version of "Revenge of the Joker" was in development, but never released.
</item 9>

<item 10>
Peter Guber
Howard Peter Guber (born March 1, 1942) is an executive, entrepreneur, educator, and author.

 He is Chairman and CEO of Mandalay Entertainment.

 Guber's most recent films from Mandalay Entertainment include "The Kids Are All Right", "Soul Surfer", and "Bernie".

 He has also produced "Batman", "The Witches of Eastwick", and "Flashdance."

 Guber's films have earned over $3 billion worldwide and 50 Academy Award nominations.
</item 10>
=== END ITEM id=f2b47a87b96e05be ===
