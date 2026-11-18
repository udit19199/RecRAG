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

=== ITEM id=74f0a027be9db2f1 ===
Gold sentences and facts:
Sentence 0:  It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of Superman called Nuclear Man.
  Fact 0.0: The film is the fourth and final film in the original Superman film series.
  Fact 0.1: The film is the first film in the original Superman series not produced by Alexander and Ilya Salkind.
  Fact 0.2: The film was produced by Golan-Globus' Cannon Films.
  Fact 0.3: The film was produced in association with Warner Bros.
  Fact 0.4: Gene Hackman returned as Lex Luthor in the film.
  Fact 0.5: Lex Luthor creates an evil solar-powered version of Superman called Nuclear Man.
Sentence 1: Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.
  Fact 1.0: Eugene Allen 'Gene' Hackman was born on January 30, 1930.
  Fact 1.1: Gene Hackman is a retired American actor.
  Fact 1.2: Gene Hackman is a retired American novelist.

Retrieved items:
<item 1>
Superman IV: The Quest for Peace
Superman IV: The Quest for Peace is a 1987 American superhero film directed by Sidney J. Furie, based on the DC Comics character Superman.

 It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but
rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of
Superman called Nuclear Man.
</item 1>

<item 2>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 2>

<item 3>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 3>

<item 4>
Lex Luthor: Man of Steel
Lex Luthor: Man of Steel (later collected as simply Luthor) is a five-issue monthly American comic book limited series written by Brian Azzarello and
illustrated by Lee Bermejo, which features Superman's nemesis Lex Luthor as the protagonist.
</item 4>

<item 5>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 5>

<item 6>
Lionel Luthor
Lionel Luthor is a fictional character portrayed by John Glover in the television series "Smallville".

 The character was initially a recurring guest in season one, and became a series regular in season two and continued until being written out of the show in
season seven.

 The character returned to the show in season ten again as a recurring guest role as a parallel universe (Earth-2) version of the character.

 In "Smallville", Lionel Luthor is the father of Lex Luthor (Michael Rosenbaum), and founder and CEO of LuthorCorp.

 Lex Luthor's father was first introduced in "Superman" comics by Jerry Siegel in 1961 and has since appeared in other Superman-related media under different
names.

 "Smallville" is the first appearance in which the character has been an intricate part of a Superman adaptation.

 Series developers Alfred Gough and Miles Millar created Lionel Luthor for "Smallville" to provide an antithesis to the parenting style of Jonathan Kent (John
Schneider) and Martha Kent (Annette O'Toole).
</item 6>

<item 7>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 7>

<item 8>
Alternative versions of Lex Luthor
The fictional character Lex Luthor has appeared in a number of media, always as the archenemy of Superman.

 Each version of the work typically establishes its own continuity, and sometimes introduces parallel universes, to the point where distinct differences in the
portrayal of the character can be identified.

 In addition, the DC Universe has been rewritten a number of times, establishing additional versions of the character.

 This article details and lists various versions of Lex Luthor depicted in works including DC Comics "Multiverse", "Elseworlds", television and film.
</item 8>

<item 9>
Lex Luthor (Smallville)
Lex Luthor is a fictional character from the television series "Smallville".

 He was a series regular from the pilot episode until the season seven finale, and has been played continuously by Michael Rosenbaum, with various actors
portraying the character as a child throughout the series.

 The character of Lex Luthor, first created for comic books by Jerry Siegel and Joe Shuster in 1940 as nemesis of Superman, was adapted to television in 2001 by
Alfred Gough and Miles Millar—this is only the third time the character has been adapted to a live action television series.

 The character has also appeared in various literature based on the "Smallville" television series, none of which directly continues from or into the television
episodes.
</item 9>

<item 10>
Gene Hackman
Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.

 In a career spanning nearly five decades, Hackman was nominated for five Academy Awards, winning Best Actor in "The French Connection" and Best Supporting
Actor in "Unforgiven".

 He won three Golden Globes and two BAFTAs.
</item 10>
=== END ITEM id=74f0a027be9db2f1 ===

=== ITEM id=7a48505b226edf7f ===
Gold sentences and facts:
Sentence 0: The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.
  Fact 0.0: The soundtrack for the 1968 film "Romeo and Juliet" was composed by Nino Rota.
  Fact 0.1: The soundtrack for the 1968 film "Romeo and Juliet" was conducted by Nino Rota.
Sentence 1: Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
  Fact 1.0: Romeo and Juliet is a 1968 British-Italian romance film.
  Fact 1.1: Romeo and Juliet is based on the tragic play of the same name.
  Fact 1.2: The play Romeo and Juliet was written between 1591 and 1595.
  Fact 1.3: The play Romeo and Juliet was written by William Shakespeare.

Retrieved items:
<item 1>
Il cappello di paglia di Firenze
Il cappello di paglia di Firenze (literally translated as "The Florentine Straw Hat" but usually titled in English language productions as "The Italian Straw
Hat") is an opera by Nino Rota to an Italian-language libretto by the composer and Ernesta Rota, based on the play "Le chapeau de paille d'Italie" by Eugène
Labiche and Marc Michel.
</item 1>

<item 2>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 2>

<item 3>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 3>

<item 4>
Graph query row:
entity: William Shakespeare
relationship: CREATED_BY
related: Romeo and Juliet
</item 4>

<item 5>
8½
8½ (Italian title: Otto e mezzo ] ) is a 1963 semi-autobiographical comedy-drama film directed by Federico Fellini.

 Co-scripted by Fellini, Tullio Pinelli, Ennio Flaiano, and Brunello Rondi, it stars Marcello Mastroianni as Guido Anselmi, a famous Italian film director.

 Shot in black-and-white by cinematographer Gianni di Venanzo, the film features a soundtrack by Nino Rota with costume and set designs by Piero Gherardi.
</item 5>

<item 6>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 6>

<item 7>
Amarcord Nino Rota
Amarcord Nino Rota is an album by various artists, recorded as a tribute to composer Nino Rota.
</item 7>

<item 8>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 8>

<item 9>
Romeo and Juliet (1968 film)
Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
</item 9>

<item 10>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 10>
=== END ITEM id=7a48505b226edf7f ===

=== ITEM id=6412640654adf204 ===
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
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 1>

<item 2>
Dying of the Light (film)
Dying of the Light is a 2014 American psychological thriller film written and directed by Paul Schrader and starring Nicolas Cage, Anton Yelchin and Irène Jacob
about a government agent who must track down and kill a terrorist before he loses his full memory from a disease.

 It was released theatrically and through VOD formats by Lionsgate on December 5, 2014.

 The film received extremely negative reviews, with controversy surrounding the heavy tampering and reediting of the footage by the studio, who denied Schrader
final-cut privilege and led him and principal members of the cast to disown the released version and campaign against it.
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
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 4>

<item 5>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 5>

<item 6>
A Star Is Born (1976 film)
A Star Is Born is a 1976 American musical drama film telling the story of a young woman, played by Barbra Streisand, who enters show business, and meets and
falls in love with an established male star, played by Kris Kristofferson, only to find her career ascending while his goes into decline.

 It is a remake of two earlier versions – the 1937 version was a drama starring Janet Gaynor and Fredric March, and the 1954 version was a musical starring Judy
Garland and James Mason.

 It will be remade for a third time in 2018 starring Stefani Germanotta and Bradley Cooper.
</item 6>

<item 7>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 7>

<item 8>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 8>

<item 9>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 9>

<item 10>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 10>
=== END ITEM id=6412640654adf204 ===

=== ITEM id=6bff94be8d32f8ac ===
Gold sentences and facts:
Sentence 0: Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".
  Fact 0.0: Stephanocereus is a genus of cactus.
  Fact 0.1: Stephanocereus is from Brazil.
  Fact 0.2: Stephanocereus is related to "Arrojadoa".
Sentence 1:  They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).
  Fact 1.0: The plants in this sentence are mostly liana-forming.
  Fact 1.1: The plants in this sentence are found in the Old World tropics.
  Fact 1.2: The plants in this sentence range from South China, Indonesia, Malaysia and the Philippines.
  Fact 1.3: The plants in this sentence range westward to Madagascar, with two species there.
  Fact 1.4: The plants in this sentence range westward to the Seychelles, with one species there.
  Fact 1.5: The plants in this sentence range southward to Australia, with three species there.
  Fact 1.6: The plants in this sentence range southward to New Caledonia, with one species there.
  Fact 1.7: The plants in this sentence range northward to India, with one species there.
  Fact 1.8: The plants in this sentence range northward to Sri Lanka, with one species there.

Retrieved items:
<item 1>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 1>

<item 2>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 2>

<item 3>
Robert Cantley
Robert Cantley is a conservationist and Managing Director of Borneo Exotics, a Sri Lankan-based plant nursery specialising in tissue-cultured and seed-grown
"Nepenthes" species and hybrids.

 Cantley has contributed to a number of papers on "Nepenthes".

 The species "Nepenthes robcantleyi", the natural hybrid "Nepenthes × cantleyi" and the cultivar "Nepenthes" 'Cantley's Red' are named after him.
</item 3>

<item 4>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 4>

<item 5>
Nepenthes × kinabaluensis
Nepenthes × kinabaluensis , or the Kinabalu Pitcher-Plant, is the natural hybrid between "N. rajah" and "N. villosa".

 It was first collected near Kambarangoh on Mount Kinabalu, Borneo by Lilian Gibbs in 1910 and later mentioned by John Muirhead Macfarlane as ""Nepenthes" sp."

 in 1914.

 Although Macfarlane did not formally name the plant, he noted that "[a]ll available morphological details suggest that this is a hybrid between "N. villosa"
and "N. rajah"".

 It was finally described in 1976 by Shigeo Kurata as "N. × kinabaluensis".

 The name was first published in "Nepenthes of Mount Kinabalu", but was a "nomen nudum" at the time as it lacked an adequate description and information on the
type specimen.

 The name was subsequently published validly by Kurata in 1984.
</item 5>

<item 6>
Nepenthes chaniana
Nepenthes chaniana ( ; after Datuk Chan Chew Lun, Managing Director of Natural History Publications (Borneo)) is a tropical pitcher plant species belonging to
the genus "Nepenthes".

 It is characterised by a dense indumentum of long, white hairs.

 Pitchers are cylindrical and mostly white to yellow in colouration.

 "Nepenthes chaniana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. boschiana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 6>

<item 7>
Nepenthes tenax
Nepenthes tenax ( ; from Latin: "tenax" "tenacious") is a lowland species of tropical pitcher plant native to northern Queensland, Australia.

 It is the third "Nepenthes" species recorded from the continent and its second endemic species.

 "Nepenthes tenax" is closely related to the two other Australian "Nepenthes" species: "N. mirabilis" and "N. rowaniae".
</item 7>

<item 8>
Nepenthes boschiana
Nepenthes boschiana ( ; after Johannes van den Bosch), or Bosch's Pitcher-Plant, is a tropical pitcher plant endemic to Borneo.

 It is most closely allied to "N. faizaliana".

 "Nepenthes borneensis" is considered a synonym of this species.

 "Nepenthes boschiana" has no known natural hybrids.

 No valid forms or varieties have been described.

 "Nepenthes boschiana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. chaniana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 8>

<item 9>
Nepenthes veitchii
Nepenthes veitchii ( or ; after James Veitch, nurseryman of the Veitch Nurseries), or Veitch's Pitcher-Plant, is a "Nepenthes" species from the island of
Borneo.

 The plant is widespread in north-western Borneo and can also be found in parts of Kalimantan.

 "Nepenthes veitchii" usually grows as an epiphyte, though the form from Bario seems to be strictly terrestrial and has not been observed to climb trees.
</item 9>

<item 10>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 10>
=== END ITEM id=6bff94be8d32f8ac ===

=== ITEM id=b597e46346596bd1 ===
Gold sentences and facts:
Sentence 0: Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway 329 (also known as the Avery Island Highway).
  Fact 0.0: Rynella is an unincorporated community.
  Fact 0.1: Rynella is in Iberia Parish, Louisiana.
  Fact 0.2: Rynella consists of a few residential dwellings.
  Fact 0.3: Rynella has a volunteer fire department.
  Fact 0.4: The residential dwellings and the volunteer fire department are along Highway 329.
  Fact 0.5: Highway 329 is also known as the Avery Island Highway.
Sentence 1:  The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA" McIlhenny (someone eventually adding an extra L to the resulting word).
  Fact 1.0: The name Rynella allegedly is an acronym.
  Fact 1.1: The acronym derives from the last two letters of the first names of three local women.
  Fact 1.2: The three local women are Rosema, Pauli, and Lei McIlhenny.
  Fact 1.3: The acronym uses the letters RY from Rosema, NE from Pauli, and LA from Lei.
  Fact 1.4: Someone eventually added an extra L to the resulting word.
Sentence 2:  They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at nearby Avery Island.
  Fact 2.0: The three women were daughters of Edward Avery McIlhenny.
  Fact 2.1: Edward Avery McIlhenny was a noted Louisiana conservationist.
  Fact 2.2: Edward Avery McIlhenny presided over McIlhenny Company.
  Fact 2.3: McIlhenny Company is the maker of Tabasco brand pepper sauce.
  Fact 2.4: McIlhenny Company is at nearby Avery Island.
Sentence 3: Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.
  Fact 3.0: Tabasco sauce is a brand of hot sauce.
  Fact 3.1: Tabasco sauce is made exclusively from tabasco peppers.
  Fact 3.2: The tabasco pepper is Capsicum frutescens var. tabasco.
Sentence 4:  "tabasco"), vinegar, and salt.
  Fact 4.0: Tabasco sauce is made with vinegar.
  Fact 4.1: Tabasco sauce is made with salt.

Retrieved items:
<item 1>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 1>

<item 2>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 2>

<item 3>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 3>

<item 4>
Valentina, made in Guadalajara from Jalisco puya chiles, is thick, mildly vinegary, and sold hot/extra-hot; Tabasco uses tabasco peppers, vinegar, and salt.
Crystal is a medium-heat Louisiana sauce exported widely. North Carolina’s Texas Pete is mild, with hotter, garlic, sriracha, and Mexican-style variants. Puerto
Rican pique steeps peppers in vinegar; Bahamian/Key West Old Sour ferments lime juice and salt
</item 4>

<item 5>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 5>

<item 6>
Tabasco sauce
Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.

 "tabasco"), vinegar, and salt.

 It is produced by McIlhenny Company of Avery Island, Louisiana.
</item 6>

<item 7>
Sriracha sauce
Sriracha (Thai: ศรีราชา ,  ] ; ) is a type of hot sauce or chili sauce made from a paste of chili peppers, distilled vinegar, garlic, sugar, and salt.

 It is named after the coastal city of Si Racha, in Chonburi Province of eastern Thailand, where it may have been first produced for dishes served at local
seafood restaurants.
</item 7>

<item 8>
Valentina (hot sauce)
Valentina is a brand of pourable hot sauce manufactured by Salsa Tamazula, a company in Guadalajara, Mexico.

 It is typically sold in 12.5-ounce and very large (one-liter or 34-ounce) glass bottles, with a flip-top cap permanently attached to the bottle (the cap does
not unscrew).

 The sauce, and the parent company's Tamazula hot sauce, are made with puya chilis from Jalisco state – similar to the Guajillo chili and known by the name
"guajillo puya" – is described as thicker than Tabasco sauce and less vinegary, with more chile flavor.

 Valentina comes in two varieties, hot and extra hot, and the sauce is known for its use on tacos and its taste, as opposed to only for its heat.

 Valentina's ingredients are water, chili peppers, vinegar, salt, spices and sodium benzoate (as a preservative).
</item 8>

<item 9>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 9>

<item 10>
Rynella, Louisiana
Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway
329 (also known as the Avery Island Highway).

 The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA"
McIlhenny (someone eventually adding an extra L to the resulting word).

 They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at
nearby Avery Island.

 (At one time McIlhenny operated a general store at Rynella and he owned much of the land that now comprises and surrounds the community.)
</item 10>
=== END ITEM id=b597e46346596bd1 ===

=== ITEM id=067b19dddb746c49 ===
Gold sentences and facts:
Sentence 0: "For the sedan variant of the car, see Volvo S40."
  Fact 0.0: The sedan variant of the car is described under Volvo S40.
Sentence 1: The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
  Fact 1.0: The Volvo S40 is a compact luxury car.
  Fact 1.1: The Volvo S40 is marketed by Volvo Cars.
  Fact 1.2: The Volvo S40 is produced by Volvo Cars.
  Fact 1.3: Volvo Cars is a Swedish manufacturer.
  Fact 1.4: The Volvo S40 was marketed and produced from 1995 to 2012.
  Fact 1.5: The Volvo S40 spanned two generations.

Retrieved items:
<item 1>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 1>

<item 2>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 2>

<item 3>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 3>

<item 4>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 4>

<item 5>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 5>

<item 6>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 6>

<item 7>
Volvo 480
The Volvo 480 is a compact car that was produced in Born, Netherlands, by Volvo from 1986 to 1995.

 It was the first front-wheel drive car made by the automaker.

 The 480 was available in only one body style on a automobile platform related to the Volvo 440/460 five door hatchback and four door sedan models.
</item 7>

<item 8>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 8>

<item 9>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 9>

<item 10>
Volvo V50
"For the sedan variant of the car, see Volvo S40."
</item 10>
=== END ITEM id=067b19dddb746c49 ===

=== ITEM id=a5ea9718f8439886 ===
Gold sentences and facts:
Sentence 0: Essence is a monthly magazine for African American women between the ages of 18 and 49.
  Fact 0.0: Essence is a monthly magazine.
  Fact 0.1: Essence is for African American women.
  Fact 0.2: Essence is aimed at women between the ages of 18 and 49.
Sentence 1: ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
  Fact 1.0: ALT for Damerne is a women's magazine.
  Fact 1.1: ALT for Damerne means 'All for the Ladies' in English.
  Fact 1.2: ALT for Damerne is written in the Danish language.
  Fact 1.3: ALT for Damerne is a weekly magazine.
  Fact 1.4: ALT for Damerne is published in Copenhagen, Denmark.

Retrieved items:
<item 1>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 1>

<item 2>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 2>

<item 3>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 3>

<item 4>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 4>

<item 5>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 5>

<item 6>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 6>

<item 7>
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
</item 7>

<item 8>
Essence Music Festival
The Essence Festival, known as "the party with a purpose", is an annual music festival which started in 1995 as a one-time event to celebrate the 25th
anniversary of "Essence", a magazine aimed primarily towards African-American women.

 It is the largest event celebrating African-American culture and music in the United States.

 Locally referred to as the Essence Fest, it has been held in New Orleans, Louisiana every year since 1994 except for 2006, when it was held in Houston, Texas
due to Hurricane Katrina's effect on New Orleans.

 It also held in Durban, KwaZulu-Natal in 2016.

 It features artists simultaneously performing on a main stage as well as four standing-room only superlounge stages.
</item 8>

<item 9>
Alt for Damerne
ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
</item 9>

<item 10>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 10>
=== END ITEM id=a5ea9718f8439886 ===

=== ITEM id=cb92a606d2080158 ===
Gold sentences and facts:
Sentence 0: Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the United States.
  Fact 0.0: Pago Pago International Airport has IATA code PPG.
  Fact 0.1: Pago Pago International Airport has ICAO code NSTU.
  Fact 0.2: Pago Pago International Airport has FAA LID PPG.
  Fact 0.3: Pago Pago International Airport is also known as Tafuna Airport.
  Fact 0.4: Pago Pago International Airport is a public airport.
  Fact 0.5: Pago Pago International Airport is located 7 miles (11.3 km) southwest of the central business district of Pago Pago.
  Fact 0.6: Pago Pago International Airport is located in the village and plains of Tafuna.
  Fact 0.7: Tafuna is on the island of Tutuila.
  Fact 0.8: Tutuila is in American Samoa.
  Fact 0.9: American Samoa is an unincorporated territory of the United States.
Sentence 1: Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business district of Hoonah, Alaska.
  Fact 1.0: Hoonah Airport has IATA code HNH.
  Fact 1.1: Hoonah Airport has ICAO code PAOH.
  Fact 1.2: Hoonah Airport has FAA LID HNH.
  Fact 1.3: Hoonah Airport is a state-owned public-use airport.
  Fact 1.4: Hoonah Airport is located one nautical mile (2 km) southeast of the central business district of Hoonah.
  Fact 1.5: Hoonah is in Alaska.

Retrieved items:
<item 1>
Inter Island Airways
Inter Island Airways (also known as "Inter Island Air") is a South Pacific regional airline based in Pago Pago, American Samoa.

 Inter Island Airways operates passenger and cargo flights in and between American Samoa, Independent Samoa and to neighboring Pacific island countries.

 Its main base of operations is at Pago Pago International Airport.
</item 1>

<item 2>
Graph query row:
airport: Hoonah Airport
connected_to: United States
relationships: ['LOCATED_IN', 'LOCATED_IN', 'PART_OF']
path: ['Hoonah Airport', 'Hoonah', 'Alaska', 'United States']
</item 2>

<item 3>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 3>

<item 4>
Tutuila
Tutuila is the largest and the main island of American Samoa in the archipelago of Samoan Islands.

 It is the third largest island in the Samoan Islands chain of the Central Pacific located roughly 4000 km northeast of Brisbane, Australia and over 1200 km
northeast of Fiji.

 It contains a large, natural harbor, Pago Pago Harbor, where Pago Pago, the capital of American Samoa is situated.

 Pago Pago International Airport is also located on Tutuila island.

 Its land expanse is about 68% of the total land area of American Samoa and with 56,000 people accounts for 95% of its population.

 The island has six terrestrial and three marine ecosystems.
</item 4>

<item 5>
Nu'uuli, American Samoa
Nu'uuli is a village on the central east coast of Tutuila Island, American Samoa.

 It is located on a peninsula several miles up from Pago Pago International Airport.
</item 5>

<item 6>
Tafuna, American Samoa
Tafuna' is a village on the east coast of Tutuila Island, American Samoa.

 It is located on a peninsula a mile north of Pago Pago International Airport and one mile south of Nu'uuli, American Samoa.
</item 6>

<item 7>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 7>

<item 8>
Veterans Memorial Stadium (Pago Pago)
Veterans Memorial Stadium is a football stadium located in Pago Pago Park, in Pago Pago, American Samoa.

 The 10,000 capacity venue is one of the smallest stadiums in Oceania, and serves as American Samoa's national stadium.

 It is the home venue of the American Samoa national football team, hosting all of their home games.

 It is currently used mostly for matches in various football codes, such as soccer, rugby league, and the territory's most popular code, American football.
</item 8>

<item 9>
WNG710
WNG710 (sometimes referred to as Pago Pago All Hazards) is a NOAA Weather Radio station that serves the entire United States territory island of American Samoa.

 It is programmed from the National Weather Service Forecast Office in Pago Pago.

 The station broadcasts weather and hazard information for the Eastern, Manu'a, and Western Districts, which are the three political divisions of American
Samoa.

 WNG710 has a sister station, WZ2529 at Mt. Olotele.

 Its transmitter is located at the opposite side of the island.
</item 9>

<item 10>
Pago Pago
Pago Pago ( ; Samoan: ] ) is the territorial capital of American Samoa.

 It is on the main island of American Samoa, Tutuila.

 The territory is served by Pago Pago International Airport at Tafuna, some 8 miles south west of Pago Pago.

 Tourism, entertainment, food, and tuna canning are its main industries.
</item 10>
=== END ITEM id=cb92a606d2080158 ===

=== ITEM id=fa894a87fdcb4973 ===
Gold sentences and facts:
Sentence 0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie, Michigan.
  Fact 0.0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant.
  Fact 0.1: The Saint Marys Falls Hydropower Plant is a Cloverland Electric Cooperative hydroelectric plant.
  Fact 0.2: The Saint Marys Falls Hydropower Plant is in Sault Ste. Marie, Michigan.
Sentence 1:  Excavation of the power canal began in September 1898 and was completed in June 1902.
  Fact 1.0: Excavation of the Edison Sault Power Canal began in September 1898.
  Fact 1.1: Excavation of the Edison Sault Power Canal was completed in June 1902.
Sentence 2:  The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
  Fact 2.0: The canal was named a Historic Civil Engineering Landmark in 1983.
  Fact 2.1: The hydroelectric complex was named a Historic Civil Engineering Landmark in 1983.
Sentence 3: The Cayuga–Seneca Canal is a canal in New York, United States.
  Fact 3.0: The Cayuga–Seneca Canal is a canal in New York, United States.
Sentence 4:  It is now part of the New York State Canal System.
  Fact 4.0: The Cayuga–Seneca Canal is now part of the New York State Canal System.

Retrieved items:
<item 1>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 1>

<item 2>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 2>

<item 3>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 3>

<item 4>
Graph query row:
first_entity: Edison Sault Power Canal
first_facts: {'name': 'Edison Sault Power Canal'}
first_connections: [{'related': 'Historic Civil Engineering Landmark', 'relationship': 'AWARDED', 'relationship_properties': {'year': 1983}}]
second_entity: Cayuga–Seneca Canal
second_facts: {'name': 'Cayuga–Seneca Canal'}
second_connections: [{'related': 'New York', 'relationship': 'LOCATED_IN', 'relationship_properties': {}}, {'related': 'New York State Canal System',
'relationship': 'PART_OF', 'relationship_properties': {}}]
direct_relationships: []
shared_connections: []
</item 4>

<item 5>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 5>

<item 6>
Power canal
A Power Canal refers to a canal used for hydraulic power generation, rather than for transport of watercraft.

 The power canal was a major factor in the Industrial revolution in New England in the 19th century.

 Most early power canals were mill races used mechanically to transfer power directly from falling water to machinery in mill buildings.

 Later, the hydraulic power generated electricity locally for the same mill factories.

 These power canals were often filled in as electricity (transported by power lines) replaced the need for local water power, and road transport needs or city
expansion needs reclaimed the land.

 Some hydraulic power canals were transformed into local electric generators, but most were closed.

 Remains of power canals can be seen in old mill towns and are often protected as historical structures today.
</item 6>

<item 7>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 7>

<item 8>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 8>

<item 9>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 9>

<item 10>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 10>
=== END ITEM id=fa894a87fdcb4973 ===

=== ITEM id=6024127cabd47c26 ===
Gold sentences and facts:
Sentence 0: Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
  Fact 0.0: The Spanish name of Teide National Park is 'Parque nacional del Teide'.
  Fact 0.1: Teide National Park is a national park.
  Fact 0.2: Teide National Park is located in Tenerife.
  Fact 0.3: Tenerife is in the Canary Islands, Spain.
Sentence 1: Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands (Spain).
  Fact 1.0: The Spanish name of Garajonay National Park is 'Parque nacional de Garajonay'.
  Fact 1.1: Garajonay National Park is located in the center of the island of La Gomera.
  Fact 1.2: Garajonay National Park is located in the north of the island of La Gomera.
  Fact 1.3: La Gomera is one of the Canary Islands in Spain.

Retrieved items:
<item 1>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 1>

<item 2>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 2>

<item 3>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 3>

<item 4>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 4>

<item 5>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 5>

<item 6>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 6>

<item 7>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 7>

<item 8>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 8>

<item 9>
Adenocarpus viscosus
Adenocarpus viscosus is a shrubby species of flowering plant in the legume family Fabaceae, subfamily Faboideae.

 It is endemic to the Canary Islands where it is known locally as Codeso del Pico.

 It can be found above 1800 m on two of the islands, La Palma in Caldera de Tabouriente and Tenerife where it is a dominant shrub in Teide National Park and
occurs in parts of Corona Forestal Nature Park and Reserva Especial de las Palomas.
</item 9>

<item 10>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 10>
=== END ITEM id=6024127cabd47c26 ===

=== ITEM id=d33ac4e03eff3363 ===
Gold sentences and facts:
Sentence 0: Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.
  Fact 0.0: Psychopsis is abbreviated Psychp in horticultural trade.
  Fact 0.1: Psychopsis is a genus of 5 known species of orchids.
  Fact 0.2: Psychopsis orchids are native to northern South America.
  Fact 0.3: Psychopsis orchids are native to Central America.
  Fact 0.4: Psychopsis orchids are native to Trinidad.
Sentence 1: Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.
  Fact 1.0: Aubrieta is commonly known as Aubretia.
  Fact 1.1: Aubrieta is a genus of about 12 species of flowering plants.
  Fact 1.2: Aubrieta is in the cabbage family Brassicaceae.

Retrieved items:
<item 1>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 1>

<item 2>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 2>

<item 3>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 3>

<item 4>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 4>

<item 5>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 5>

<item 6>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 6>

<item 7>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 7>

<item 8>
Aubrieta
Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.

 The genus is named after Claude Aubriet, a French flower-painter.

 It originates from southern Europe east to central Asia but is now a common garden escape throughout Europe.

 It is a low, spreading plant, hardy, evergreen and perennial, with small violet, pink or white flowers, and inhabits rocks and banks.

 It prefers light, well-drained soil, is tolerant of a wide pH range, and can grow in partial shade or full sun.
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 10>
=== END ITEM id=d33ac4e03eff3363 ===

=== ITEM id=50a90a8842776e8b ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.0: The Sports Museum is also known as The Sports Museum of New England.
  Fact 0.1: The Sports Museum is a non-profit museum.
  Fact 0.2: The Sports Museum is currently located in the TD Garden.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.
Sentence 1: TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.
  Fact 1.0: TD Garden is often called Boston Garden.
  Fact 1.1: TD Garden is often called "The Garden".
  Fact 1.2: TD Garden is a multi-purpose arena.
  Fact 1.3: TD Garden is in Boston, Massachusetts.
Sentence 2:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.0: TD Garden opened in 1995.
  Fact 2.1: TD Garden opened as a replacement for the original Boston Garden.
  Fact 2.2: TD Garden has been known as Shawmut Center.
  Fact 2.3: TD Garden has been known as FleetCenter.
  Fact 2.4: TD Garden has been known as TD Banknorth Garden.

Retrieved items:
<item 1>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 1>

<item 2>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 2>

<item 3>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 3>

<item 4>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 4>

<item 5>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 5>

<item 6>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 6>

<item 7>
The entries describe sports teams, venues, parks, and communities: Seattle Sounders FC 2, a Sounders reserve team founded in 2014; Charlotte Hornets; London
Skolars; Sacramento Republic FC, 2014 USL champions seeking MLS expansion; and several baseball, racing, and pipe-band organizations. Facilities include
Boston’s TD Garden, Braves Field site, and Camden Yards complex; BYU’s Smith Fieldhouse; Campeche’s Estadio Nelson Barrera; and Sacramento’s Papa
</item 7>

<item 8>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 8>

<item 9>
- **J. Robert Donnelly Husky Heritage Sports Museum:** UConn campus museum opened 2002; celebrates Huskies athletics with trophies, memorabilia, video displays,
and a basketball championship exhibit.
- **Jeonju National Museum:** South Korean national museum opened in 1990.
- **Croatian Natural History Museum:** Zagreb institution founded in 1846; Croatia’s largest natural-history museum, with over 2 million items and a public
scientific library.
- **National
</item 9>

<item 10>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 10>
=== END ITEM id=50a90a8842776e8b ===
