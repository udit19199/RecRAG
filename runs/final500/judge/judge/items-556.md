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

=== ITEM id=410efb51d35068af ===
Gold sentences and facts:
Sentence 0:  Editors are based in Silver Spring, Maryland and Seoul, Korea.
  Fact 0.0: Editors are based in Silver Spring, Maryland.
  Fact 0.1: Editors are based in Seoul, Korea.
Sentence 1:  The magazine is headquartered in Baltimore, Maryland.
  Fact 1.0: The magazine is headquartered in Baltimore, Maryland.

Retrieved items:
<item 1>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 1>

<item 2>
Ángel Manuel Rodríguez
Ángel Manuel Rodríguez (1945—) is a Seventh-day Adventist theologian and was the director of the Biblical Research Institute (BRI) before his retirement.

 His special research interests include Old Testament, Sanctuary and Atonement, and Old Testament Theology.

 He has written several books, and authors a monthly column in "Adventist World".
</item 2>

<item 3>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 3>

<item 4>
Adventist World
Adventist World is a monthly international magazine of the Seventh-day Adventist Church published by the Review and Herald Publishing Association.

 Editors are based in Silver Spring, Maryland and Seoul, Korea.
</item 4>

<item 5>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 5>

<item 6>
Seventh-day Adventist education
The Seventh-day Adventist educational system is part of the Seventh-day Adventist Church and overseen by the General Conference of Seventh-day Adventist located
in Silver Spring, Maryland.The educational system is the second-largest Christian school-system in the world, after the Roman Catholic system.
</item 6>

<item 7>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 7>

<item 8>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 8>

<item 9>
Adventist HealthCare
Adventist HealthCare is a not-for-profit health services organization based in Gaithersburg, Maryland that employs more than 6,200 people and provides
healthcare for more than 400,000 individuals in the community each year.

 The primary service area for Adventist HealthCare is the Washington, D.C. metropolitan area.

 Despite similar names, it is not a part of the California-based Adventist Health, or the Florida-based Adventist Health System.
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
=== END ITEM id=410efb51d35068af ===

=== ITEM id=93033bd3ad2948be ===
Gold sentences and facts:
Sentence 0:  He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.
  Fact 0.0: The writer has written three novels.
  Fact 0.1: The writer has written three collections of short stories.
  Fact 0.2: The writer has written a collection of novellas.
  Fact 0.3: The writer has written seven volumes of essays.
  Fact 0.4: Three of the writer's essay volumes have won National Book Critics Circle Award prizes.
  Fact 0.5: One of the writer's essay volumes, 'A Temple of Texts' (2006), won the Truman Capote Award for Literary Criticism.
Sentence 1: Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for Literature.
  Fact 1.0: Naguib Mahfouz was born on December 11, 1911.
  Fact 1.1: Naguib Mahfouz died on August 30, 2006.
  Fact 1.2: Naguib Mahfouz's Arabic name is نجيب محفوظ, transliterated Nagīb Maḥfūẓ.
  Fact 1.3: Naguib Mahfouz was an Egyptian writer.
  Fact 1.4: Naguib Mahfouz won the 1988 Nobel Prize for Literature.

Retrieved items:
<item 1>
Naguib Mahfouz
Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for
Literature.

 He is regarded as one of the first contemporary writers of Arabic literature, along with Tawfiq el-Hakim, to explore themes of existentialism.

 He published 34 novels, over 350 short stories, dozens of movie scripts, and five plays over a 70-year career.

 Many of his works have been made into Egyptian and foreign films.
</item 1>

<item 2>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 2>

<item 3>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 3>

<item 4>
The Beginning and the End (novel)
The Beginning and the End (بداية ونهاية) is a novel by Naguib Mahfouz, who was awarded the Nobel Prize in literature in 1988.

 An Egyptian, Naguib has been credited with modernizing Arabic literature, with his prolific writing style and his themes on existentialism.

 The novel is marked by very bold characterization by Naguib Mahfouz, for the time period that the story is set in.

 The story moves at a prolific pace and tries to look at the world from each characters view point.
</item 4>

<item 5>
Naguib Mahfouz Medal for Literature
The Naguib Mahfouz Medal for Literature is a literary award for Arabic literature.

 It is given to the best contemporary novel written in Arabic, but not available in English translation.

 The winning book is then translated into English, and published by American University in Cairo Press.

 It was first awarded in 1996 and is presented annually on December 11, the birthday of Nobel laureate Naguib Mahfouz, by the President of the American
University in Cairo.
</item 5>

<item 6>
Graph query row:
person: Naguib Mahfouz
relationship: AWARDED
prize: Nobel Prize in Literature
year: 1988
years: None
</item 6>

<item 7>
Thebes at War
Thebes at War is an early novel by the Egyptian writer Naguib Mahfouz.

 It was originally published in Arabic in 1944.

 An English translation by Humphrey Davies appeared in 2003.

 The novel is one of several that Mahfouz wrote at the beginning of his career, with Pharaonic Egypt as their setting.

 Others in this series of novels include "Khufu's Wisdom" (1939) and "Rhadopis of Nubia" (1943).

 All have been translated into English and appeared in one volume under the title "Three Novels of Ancient Egypt" (Everyman's Library, 2007).
</item 7>

<item 8>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 8>

<item 9>
Cairo Trilogy
The Cairo Trilogy (Arabic: الثلاثية‎ ‎ ("The Trilogy") or ثلاثية القاهرة ("The Cairo Trilogy")) is a trilogy of novels written by the Egyptian novelist and
Nobel Prize winner Naguib Mahfouz, and one of the prime works of his literary career.
</item 9>

<item 10>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 10>
=== END ITEM id=93033bd3ad2948be ===

=== ITEM id=6e3268d79c46e71f ===
Gold sentences and facts:
Sentence 0:  It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.
  Fact 0.0: The work is the third of Gluck's so-called reform operas for Vienna.
  Fact 0.1: The work follows 'Alceste' and 'Orfeo ed Euridice' among Gluck's reform operas for Vienna.
  Fact 0.2: The work is the least often performed of the three reform operas for Vienna.
Sentence 1: Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of Orpheus, set to a libretto by Ranieri de' Calzabigi.
  Fact 1.0: Orfeo ed Euridice is an opera composed by Christoph Willibald Gluck.
  Fact 1.1: Orfeo ed Euridice is based on the myth of Orpheus.
  Fact 1.2: Orfeo ed Euridice is set to a libretto by Ranieri de' Calzabigi.
  Fact 1.3: The French version of Orfeo ed Euridice is titled Orphée et Eurydice.
  Fact 1.4: The English title of Orfeo ed Euridice is 'Orpheus and Eurydice'.
Sentence 2:  The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.
  Fact 2.0: The piece was first performed at the Burgtheater in Vienna on 5 October 1762.
  Fact 2.1: The first performance of the piece was in the presence of Empress Maria Theresa.

Retrieved items:
<item 1>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 1>

<item 2>
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 2>

<item 3>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 3>

<item 4>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 4>

<item 5>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 5>

<item 6>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 6>

<item 7>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 7>

<item 8>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 8>

<item 9>
Maria Theresa (film)
Maria Theresa (German: Maria Theresia) is a 1951 Austrian historical drama film directed by  Emil E. Reinert and starring  Paula Wessely, Fred Liewehr and
Marianne Schönauer.

 It portrays the life of the eighteenth century Habsburg Empress Maria Theresa.
</item 9>

<item 10>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 10>
=== END ITEM id=6e3268d79c46e71f ===

=== ITEM id=c377dfb3606eab5f ===
Gold sentences and facts:
Sentence 0:  One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.
  Fact 0.0: A nunatak was first roughly mapped by the Second German Antarctic Expedition of 1911–12
  Fact 0.1: The Second German Antarctic Expedition of 1911–12 was under Wilhelm Filchner
  Fact 0.2: The Second German Antarctic Expedition named the nunatak Moltke Nunatak
Sentence 1:  He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.
  Fact 1.0: The nunatak was named for General Helmuth von Moltke
  Fact 1.1: General Helmuth von Moltke was Chief of the German General Staff
  Fact 1.2: General Helmuth von Moltke was Secretary of State for Home Affairs
Sentence 2: Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal) Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.
  Fact 2.0: Helmuth Johann Ludwig von Moltke was born 23 May 1848
  Fact 2.1: Helmuth Johann Ludwig von Moltke died 18 June 1916
  Fact 2.2: Helmuth Johann Ludwig von Moltke is also known as Moltke the Younger
  Fact 2.3: Helmuth Johann Ludwig von Moltke was a nephew of Generalfeldmarschall (Field Marshal) Helmuth Karl Bernhard von Moltke
  Fact 2.4: Helmuth Johann Ludwig von Moltke served as Chief of the German General Staff from 1906 to 1914

Retrieved items:
<item 1>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 1>

<item 2>
Schlieffen Plan
The Schlieffen Plan (German: "Schlieffen-Plan" , ] ) was the name given after World War I to the thinking behind the German invasion of France and Belgium on 4
August 1914.

 Field Marshal Alfred von Schlieffen, the Chief of the Imperial Army German General Staff from 1891 to 1906, devised a deployment plan for a war-winning
offensive, in a one-front war against the French Third Republic from 1905–06.

 After the war, the German official historians of the "Reichsarchiv" and other writers, described the plan as a blueprint for victory.

 German historians claimed that the plan had been ruined by "Generaloberst" (Colonel-General) Helmuth von Moltke the Younger, the Commander-in-Chief of the
German army after Schlieffen retired in 1906, who was dismissed after the First Battle of the Marne (5–12 September 1914).
</item 2>

<item 3>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 3>

<item 4>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 4>

<item 5>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 5>

<item 6>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 6>

<item 7>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 7>

<item 8>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 8>

<item 9>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 9>

<item 10>
Moltke Nunataks
The Moltke Nunataks ( ) are a chain of north–south trending nunataks close to the northeastern end of the Filchner Ice Shelf, Antarctica.

 One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.

 He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.

 Surveys during the mid-1950s by British, Argentine and United States expeditions indicate that a group of four or five nunataks exist in the area.
</item 10>
=== END ITEM id=c377dfb3606eab5f ===

=== ITEM id=f58c0cdfeaf95b9c ===
Gold sentences and facts:
Sentence 0: Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist, pianist, and primary songwriter for the Christian rock band Relient K.
  Fact 0.0: Matthew Arnold 'Matt' Thiessen was born August 12, 1980.
  Fact 0.1: Matt Thiessen is a Canadian-American musician, singer, and songwriter.
  Fact 0.2: Matt Thiessen is known for being co-founder of the Christian rock band Relient K.
  Fact 0.3: Matt Thiessen is known for being lead singer of the Christian rock band Relient K.
  Fact 0.4: Matt Thiessen is known for being guitarist of the Christian rock band Relient K.
  Fact 0.5: Matt Thiessen is known for being pianist of the Christian rock band Relient K.
  Fact 0.6: Matt Thiessen is known for being primary songwriter for the Christian rock band Relient K.
Sentence 1: George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.
  Fact 1.0: George Thorogood was born February 24, 1950.
  Fact 1.1: George Thorogood is an American musician, singer, and songwriter.
  Fact 1.2: George Thorogood is from Wilmington, Delaware.

Retrieved items:
<item 1>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 1>

<item 2>
Better Than the Rest
Better Than the Rest is the third album (mini-album) of songs by George Thorogood and the Destroyers, recorded in 1974 and released in 1979.

 At the time the songs were recorded, Thorogood was an unknown artist who was not signed to a record label.

 In 1979, after Thorogood had released his first two official albums on Rounder Records, "Better Than The Rest" was released by MCA.

 In 1986, the songs from this album were released on compact disc with the title "Nadine" using a different track order.
</item 2>

<item 3>
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 3>

<item 4>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 4>

<item 5>
More George Thorogood and The Destroyers
More George Thorogood and the Destroyers is the fourth album by George Thorogood and The Destroyers, released in 1980.

 An alternate name for the album is I'm Wanted.
</item 5>

<item 6>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 6>

<item 7>
George Thorogood and the Destroyers (album)
George Thorogood and the Destroyers is the self-titled debut album by American blues rock band George Thorogood and the Destroyers, released in 1977.

 Consisting mostly of covers of blues hits, it included a medley of John Lee Hooker's "House Rent Boogie" and "One Bourbon, One Scotch, One Beer", the latter a
song written by Rudy Toombs for Amos Milburn, and later covered by Hooker.
</item 7>

<item 8>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 8>

<item 9>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 9>

<item 10>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 10>
=== END ITEM id=f58c0cdfeaf95b9c ===

=== ITEM id=ba52d90c8f2dc3c7 ===
Gold sentences and facts:
Sentence 0: Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.
  Fact 0.0: Jerome John Garcia was born August 1, 1942.
  Fact 0.1: Jerome John Garcia died August 9, 1995.
  Fact 0.2: Jerome John Garcia was an American singer-songwriter.
  Fact 0.3: Jerome John Garcia was an American guitarist.
  Fact 0.4: Jerome John Garcia was best known for his work as the lead guitarist of the Grateful Dead.
  Fact 0.5: Jerome John Garcia was best known for his work as a vocalist with the Grateful Dead.
  Fact 0.6: The Grateful Dead came to prominence during the counterculture era in the 1960s.
Sentence 1: Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
  Fact 1.0: Joe Gooch was born 3 May 1977.
  Fact 1.1: Joe Gooch was Ten Years After's most recent lead vocalist.
  Fact 1.2: Joe Gooch was Ten Years After's most recent lead guitarist.

Retrieved items:
<item 1>
The Very Best of Jerry Garcia
The Very Best of Jerry Garcia is a compilation album with recordings by Jerry Garcia, Grateful Dead, Jerry Garcia Band, Old and in the Way, Jerry Garcia
Acoustic Band, and Reconstruction.

 It is meant to showcase Jerry's non-Grateful Dead work.
</item 1>

<item 2>
Cats Under the Stars
Cats Under the Stars is an album by the Jerry Garcia Band.

 Released in 1978, it is the only studio album credited to the band.

 It was Jerry Garcia's first album on the Arista label and his first LP using the band name Jerry Garcia Band.

 It includes only one song that was performed by the Grateful Dead.

 Keith and Donna Godchaux, who were at the time members of the Garcia Band, contributed to the music.
</item 2>

<item 3>
On Broadway: Act One – October 28th, 1987
On Broadway: Act One – October 28th, 1987 is a three-CD live album by the Jerry Garcia Band and the Jerry Garcia Acoustic Band, two music groups led by Grateful
Dead guitarist and singer Jerry Garcia.

 It contains three complete sets of music, recorded at the Lunt-Fontanne Theatre in New York City on October 28, 1987.

 It was released by ATO Records on June 23, 2015.
</item 3>

<item 4>
Sugaree
"Sugaree" is a song written by long-time Grateful Dead lyricist Robert Hunter and composed by guitarist Jerry Garcia.

 It was written for Jerry Garcia's first solo album "Garcia", which was released in January 1972.

 As with the songs on the rest of the album, Garcia plays every instrument himself (except drums, played by Bill Kreutzmann), including acoustic guitar, bass
guitar, and an electric guitar played through a Leslie speaker.
</item 4>

<item 5>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 5>

<item 6>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 6>

<item 7>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 7>

<item 8>
Garcia Plays Dylan
Garcia Plays Dylan is an album composed of various live performances featuring Jerry Garcia playing covers of Bob Dylan songs.

 It is culled from performances from 1973–1995, and features Garcia playing with Grateful Dead, Legion of Mary, Jerry Garcia Band, and Garcia-Saunders.

 Garcia takes lead vocals on all tracks.
</item 8>

<item 9>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 9>

<item 10>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 10>
=== END ITEM id=ba52d90c8f2dc3c7 ===

=== ITEM id=4a561963f04eecd5 ===
Gold sentences and facts:
Sentence 0:  It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
  Fact 0.0: The series was created by Jay Lender.
  Fact 0.1: The series was created by Micah Wright.
  Fact 0.2: Jay Lender was a former writer for The Angry Beavers.
  Fact 0.3: Jay Lender was a former writer for SpongeBob SquarePants.
  Fact 0.4: Jay Lender was a former writer for Phineas and Ferb.
  Fact 0.5: Micah Wright was a former writer for The Angry Beavers.
  Fact 0.6: Micah Wright was a former writer for SpongeBob SquarePants.
  Fact 0.7: Micah Wright was a former writer for Phineas and Ferb.
Sentence 1:  The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.
  Fact 1.0: The series chronicles the adventures and endeavors of the title character and his various friends.
  Fact 1.1: The series is set in the fictional underwater city of Bikini Bottom.

Retrieved items:
<item 1>
SpongeBob SquarePants: Battle for Bikini Bottom
SpongeBob SquarePants: Battle for Bikini Bottom is a 2003 video game based on the animated series of the same name, developed by Heavy Iron Studios, AWE Games,
and Vicarious Visions, and published by THQ.

 The game was released for the PlayStation 2, Game Boy Advance (GBA), Xbox, and Nintendo GameCube consoles as well as a port for Microsoft Windows.

 The Microsoft Windows version of the game is a point-and-click adventure game, whereas the 6th generation platforms received a 3D platformer.

 The Game Boy Advance version received a 2D platformer.

 All versions of the game feature an original storyline, in which the player attempts to defend Bikini Bottom from an invasion of robots created by Plankton
with a machine called the Duplicatotron 3000, playing as SpongeBob in all versions, as well as Patrick and Sandy in the console versions.

 The game was released on October 31, 2003, in North America and in Europe on November 28, 2003.
</item 1>

<item 2>
Rock Bottom (SpongeBob SquarePants)
"Rock Bottom" is the 34th episode of the first season of the American animated television series "SpongeBob SquarePants".

 It originally was produced in 1999 and aired on Nickelodeon in the United States on March 15, 2000.

 The series follows the adventures of the title character in the underwater city of Bikini Bottom.

 In the episode, SpongeBob becomes stranded in a abyssopelagic zone called Rock Bottom.
</item 2>

<item 3>
The material surveys animation-related games, series, episodes, seasons, and creators. Highlights include *Cars 2: The Video Game* (2011), Disney racing titles,
and *Battle for Bikini Bottom* (2003); *SpongeBob SquarePants*, its seasons, and episodes such as “Band Geeks,” “Rock Bottom,” and “SpongeBob, You’re Fired”;
and Nickelodeon’s *Harvey Beaks* (2015). It also profiles voice
</item 3>

<item 4>
List of SpongeBob SquarePants episodes
"SpongeBob SquarePants" is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series is set in the fictional underwater city of Bikini Bottom, and centers on the adventures and endeavors of SpongeBob SquarePants, an over-optimistic
sea sponge that annoys other characters.

 Many of the ideas for the show originated in an unpublished, educational comic book titled "The Intertidal Zone", which Hillenburg created in the mid-1980s.

 He began developing "SpongeBob SquarePants" into a television series in 1996 after the cancellation of "Rocko's Modern Life", another Nickelodeon television
series which Hillenburg previously directed.
</item 4>

<item 5>
Robocalypse
Robocalypse is a real-time strategy video game developed and published by Vogster Entertainment for the Nintendo DS handheld video game console.

 It has only been released in North America.

 It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
</item 5>

<item 6>
SpongeBob SquarePants
SpongeBob SquarePants is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The series' popularity has made it a media franchise, as well as the highest rated series to ever air on Nickelodeon, and the most distributed property of MTV
Networks.

 As of 2015, the media franchise has generated $12 billion in merchandising revenue for Nickelodeon.
</item 6>

<item 7>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 7>

<item 8>
SpongeBob SquarePants (season 4)
The fourth season of the American animated television series "SpongeBob SquarePants", created by former marine biologist and animator Stephen Hillenburg, aired
on Nickelodeon from May 6, 2005 to July 24, 2007, and contained 20 episodes, beginning with the episodes "Fear of a Krabby Patty" and "Shell of a Man".

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, while writer Paul Tibbitt acted as the supervising/co-executive producer and showrunner.

 The show underwent a hiatus on television as Hillenburg halted the production in 2002 to work on the film adaptation of the series, "The SpongeBob SquarePants
Movie".

 Once the film was finalized and the previous season had completed broadcast on television, Hillenburg wanted to end the show, but the success of the series led
to more episodes, so Tibbitt took over Hillenburg's position as showrunner and began working on a fourth season for broadcast in 2005.
</item 8>

<item 9>
Legends of Bikini Bottom
Legends of Bikini Bottom is an anthology series of six episodes in the American animated television series "SpongeBob SquarePants", as part of its seventh
season.

 As the name suggests, the episodes have plots involving things like monsters, magic and mythical creatures.

 The first episode called "Trenchbillies" first premiered on the online social networking service Facebook on January 27, 2011 before airing on Nickelodeon on
January 29, 2011.

 The four other episodes titled "Sponge-Cano!"

, "The Main Drain", "The Monster Who Came to Bikini Bottom" and "Welcome to the Bikini Bottom Triangle" premiered on Nickelodeon as part of an hour-long special
on January 28, 2011.

 The decision to air the series online was aimed at attracting "the young and the restless hooked to the internet and the social media."
</item 9>

<item 10>
SpongeBob SquarePants (season 3)
The third season of the American animated television series "SpongeBob SquarePants", created by Stephen Hillenburg, aired on Nickelodeon from October 5, 2001 to
October 11, 2004, and consists of 20 episodes.

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, who also acted as the showrunner.

 Hillenburg halted production on the show to work on the 2004 film adaptation of the series, "The SpongeBob SquarePants Movie".

 After production on the film, Hillenburg resigned from the show as its showrunner, and appointed staff writer Paul Tibbitt to overtake the position.

 Season 3 was originally set to end the series after the release of the film, but the success prevented the series from ending, leading to a fourth season.
</item 10>
=== END ITEM id=4a561963f04eecd5 ===

=== ITEM id=03c29c57e20eda47 ===
Gold sentences and facts:
Sentence 0: Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.
  Fact 0.0: Aleksandr Ivanovsky lived from 1881 to 1968.
  Fact 0.1: Aleksandr Ivanovsky was a Russian screenwriter.
  Fact 0.2: Aleksandr Ivanovsky was a Russian film director.
  Fact 0.3: Aleksandr Ivanovsky worked during the Soviet era.
Sentence 1: Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera" (1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".
  Fact 1.0: Arthur Lubin was born on July 25, 1898.
  Fact 1.1: Arthur Lubin died on May 12, 1995.
  Fact 1.2: Arthur Lubin was an American film director.
  Fact 1.3: Arthur Lubin was an American producer.
  Fact 1.4: Arthur Lubin directed several 'Abbott & Costello' films.
  Fact 1.5: Arthur Lubin directed 'Phantom of the Opera', released in 1943.
  Fact 1.6: Arthur Lubin directed the 'Francis the Talking Mule' series.
  Fact 1.7: Arthur Lubin created the talking-horse TV series 'Mister Ed'.

Retrieved items:
<item 1>
Aleksei Petrenko
Aleksei Vasilyevich Petrenko (Russian: Алексей Васильевич Петренко ; 26 March 1938 – 22 February 2017) was a Soviet and Russian film and stage actor.

 He played Grigori Rasputin in the Elem Klimov's historical drama
</item 1>

<item 2>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 2>

<item 3>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 3>

<item 4>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
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
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 6>

<item 7>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 7>

<item 8>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 8>

<item 9>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 9>

<item 10>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 10>
=== END ITEM id=03c29c57e20eda47 ===

=== ITEM id=992ed635c911391b ===
Gold sentences and facts:
Sentence 0: The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.
  Fact 0.0: The Shame of a City is a 2006 feature-length documentary.
  Fact 0.1: The Shame of a City premiered at the Philadelphia Film Festival.
Sentence 1: Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and preserved in the WPA Slave Narrative Collection.
  Fact 1.0: Unchained Memories is a 2003 documentary film.
  Fact 1.1: Unchained Memories is about the stories of former slaves.
  Fact 1.2: The former slaves were interviewed during the 1930s.
  Fact 1.3: The interviews were part of the Federal Writers' Project.
  Fact 1.4: The stories are preserved in the WPA Slave Narrative Collection.

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
Beautiful Darling
Beautiful Darling: The Life and Times of Candy Darling, Andy Warhol Superstar is a 2010 feature-length documentary film about Candy Darling, the transgender
pioneer, actress and Andy Warhol Superstar.

 The film was written and directed by James Rasin and features Chloë Sevigny as "the voice of Candy Darling", reading from Candy's private diaries and letters.

 Patton Oswalt voices Andy Warhol and Truman Capote.

 Louis Durra composed the score.
</item 2>

<item 3>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 3>

<item 4>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 4>

<item 5>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 5>

<item 6>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 6>

<item 7>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 7>

<item 8>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 8>

<item 9>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 9>

<item 10>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 10>
=== END ITEM id=992ed635c911391b ===

=== ITEM id=a3df7935280490e8 ===
Gold sentences and facts:
Sentence 0: The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
  Fact 0.0: The Consul is an opera.
  Fact 0.1: The Consul is an opera in three acts.
  Fact 0.2: The Consul has music by Gian Carlo Menotti.
  Fact 0.3: The Consul has a libretto by Gian Carlo Menotti.
  Fact 0.4: The Consul is Gian Carlo Menotti's first full-length opera.
Sentence 1:  He completed the music for the opera while living in Zurich in 1916.
  Fact 1.0: Gian Carlo Menotti completed the music for The Consul.
  Fact 1.1: Gian Carlo Menotti completed the music for The Consul in 1916.
  Fact 1.2: Gian Carlo Menotti was living in Zurich in 1916.
  Fact 1.3: Gian Carlo Menotti completed the music for The Consul while living in Zurich.

Retrieved items:
<item 1>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 1>

<item 2>
The Consul
The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
</item 2>

<item 3>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 3>

<item 4>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 4>

<item 5>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 5>

<item 6>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 6>

<item 7>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 7>

<item 8>
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 8>

<item 9>
Caisson lock
A caisson lock was invented in the late 18th century as a solution to minimise the use of great volumes of water required to raise and lower canal boats through
large height differences.

 It was normal to only raise and lower boats through small height differences of a few feet when traversing undulating terrain.

 A solution was required when either large height differences were encountered or water was in short supply.

 The Caisson (or Caisoon) was thought to be one solution.

 The technology of the time was not capable of achieving this type of construction economically with current building materials.

 This design attempted was a type of canal lock in which a narrowboat is floated into a sealed watertight box and raised or lowered between two different canal
water levels.

 It was designed primarily as a water-saving measure, and also was an attempt to minimise construction costs compared with other engineering solutions of the
time.

 In use it was capable of replacing up to seven conventional locks.

 Other design benefits were speed of boat descent/ascent, and only a little loss of water when operating compared with a conventional boat lock.
</item 9>

<item 10>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 10>
=== END ITEM id=a3df7935280490e8 ===

=== ITEM id=e9f3af61c35c2099 ===
Gold sentences and facts:
Sentence 0: The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.
  Fact 0.0: The Basset Bleu de Gascogne is also known as the Blue Gascony Basset.
  Fact 0.1: The Basset Bleu de Gascogne is a long-backed, short-legged breed of dog.
  Fact 0.2: The Basset Bleu de Gascogne is a dog of the hound type.
Sentence 1:  A French native breed, it is rare outside of its homeland.
  Fact 1.0: The Basset Bleu de Gascogne is a French native breed.
  Fact 1.1: The Basset Bleu de Gascogne is rare outside of its homeland.
Sentence 2: The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.
  Fact 2.0: The Bulldog is a medium-sized breed of dog.
  Fact 2.1: The Bulldog is commonly referred to as the English Bulldog.
  Fact 2.2: The Bulldog is commonly referred to as the British Bulldog.

Retrieved items:
<item 1>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 1>

<item 2>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 2>

<item 3>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 3>

<item 4>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 4>

<item 5>
Basset Bleu de Gascogne
The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.

 The breed originated in the Middle Ages, descended from the Grand Bleu de Gascogne.

 It nearly became extinct around the early 19th century; its salvation was attributed to one Alain Bourbon.

 A French native breed, it is rare outside of its homeland.

 It is recognized internationally by the Fédération Cynologique Internationale, in the UK by The Kennel Club, and by the United Kennel Club in the United
States.

 The "bleu" of its name is a reference to its coat which has a ticked appearance.
</item 5>

<item 6>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 6>

<item 7>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 7>

<item 8>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
</item 8>

<item 9>
Ariegeois
The Ariegeois is a breed of dog from the "département" of Ariège in the Midi-Pyrenées region of southern France.

 It is a medium-sized pack-hunting scenthound deriving from crossing of Grand Bleu de Gascogne and Grand Gascon-Saintongeois hounds with local Briquet dogs.

 It is used both as a courser and for driving game to waiting guns.

 While most successful with hares, it is also used for hunting deer and boar.

 It is distinguished by its friendly nature with other hounds and affection for human companions.
</item 9>

<item 10>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 10>
=== END ITEM id=e9f3af61c35c2099 ===

=== ITEM id=ff1c5456efddd83c ===
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
Pago Pago
Pago Pago ( ; Samoan: ] ) is the territorial capital of American Samoa.

 It is on the main island of American Samoa, Tutuila.

 The territory is served by Pago Pago International Airport at Tafuna, some 8 miles south west of Pago Pago.

 Tourism, entertainment, food, and tuna canning are its main industries.
</item 1>

<item 2>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 2>

<item 3>
Tafuna, American Samoa
Tafuna' is a village on the east coast of Tutuila Island, American Samoa.

 It is located on a peninsula a mile north of Pago Pago International Airport and one mile south of Nu'uuli, American Samoa.
</item 3>

<item 4>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 4>

<item 5>
Tutuila
Tutuila is the largest and the main island of American Samoa in the archipelago of Samoan Islands.

 It is the third largest island in the Samoan Islands chain of the Central Pacific located roughly 4000 km northeast of Brisbane, Australia and over 1200 km
northeast of Fiji.

 It contains a large, natural harbor, Pago Pago Harbor, where Pago Pago, the capital of American Samoa is situated.

 Pago Pago International Airport is also located on Tutuila island.

 Its land expanse is about 68% of the total land area of American Samoa and with 56,000 people accounts for 95% of its population.

 The island has six terrestrial and three marine ecosystems.
</item 5>

<item 6>
WNG710
WNG710 (sometimes referred to as Pago Pago All Hazards) is a NOAA Weather Radio station that serves the entire United States territory island of American Samoa.

 It is programmed from the National Weather Service Forecast Office in Pago Pago.

 The station broadcasts weather and hazard information for the Eastern, Manu'a, and Western Districts, which are the three political divisions of American
Samoa.

 WNG710 has a sister station, WZ2529 at Mt. Olotele.

 Its transmitter is located at the opposite side of the island.
</item 6>

<item 7>
Veterans Memorial Stadium (Pago Pago)
Veterans Memorial Stadium is a football stadium located in Pago Pago Park, in Pago Pago, American Samoa.

 The 10,000 capacity venue is one of the smallest stadiums in Oceania, and serves as American Samoa's national stadium.

 It is the home venue of the American Samoa national football team, hosting all of their home games.

 It is currently used mostly for matches in various football codes, such as soccer, rugby league, and the territory's most popular code, American football.
</item 7>

<item 8>
Inter Island Airways
Inter Island Airways (also known as "Inter Island Air") is a South Pacific regional airline based in Pago Pago, American Samoa.

 Inter Island Airways operates passenger and cargo flights in and between American Samoa, Independent Samoa and to neighboring Pacific island countries.

 Its main base of operations is at Pago Pago International Airport.
</item 8>

<item 9>
Pan Am Flight 806
Pan Am Flight 806 was an international scheduled flight from Auckland, New Zealand, to Los Angeles, California, with intermediate stops at Pago Pago, American
Samoa and Honolulu, Hawaii.

 On January 30, 1974, the Boeing 707 "Clipper Radiant" crashed on approach to Pago Pago International Airport, killing 87 passengers and ten crew members.
</item 9>

<item 10>
Nu'uuli, American Samoa
Nu'uuli is a village on the central east coast of Tutuila Island, American Samoa.

 It is located on a peninsula several miles up from Pago Pago International Airport.
</item 10>
=== END ITEM id=ff1c5456efddd83c ===
