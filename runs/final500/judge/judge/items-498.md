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

=== ITEM id=f6314516b3f08974 ===
Gold sentences and facts:
Sentence 0: Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on science and technology subjects.
  Fact 0.0: Popular Science is also known as PopSci.
  Fact 0.1: Popular Science is an American bi-monthly magazine.
  Fact 0.2: Popular Science carries popular science content.
  Fact 0.3: Popular science content refers to articles for the general reader on science and technology subjects.
Sentence 1: Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.
  Fact 1.0: Girlfriends was a women's magazine.
  Fact 1.1: Girlfriends provided critical coverage of culture.
  Fact 1.2: Girlfriends provided critical coverage of entertainment.
  Fact 1.3: Girlfriends provided critical coverage of world events.
  Fact 1.4: Girlfriends' coverage was from a lesbian perspective.
Sentence 2:  Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.
  Fact 2.0: Girlfriends was published monthly from San Francisco since 1993.
  Fact 2.1: Girlfriends was distributed nationwide by Disticor.
Sentence 3:  "Girlfriends" magazine ceased publication in 2006.
  Fact 3.0: Girlfriends magazine ceased publication in 2006.

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
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 2>

<item 3>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 3>

<item 4>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 4>

<item 5>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 5>

<item 6>
Popular science
Popular science (also pop-science or popsci) is interpretation of science intended for a general audience.

 While science journalism focuses on recent scientific developments, popular science is more broad-ranging.

 It may be written by professional science journalists or by scientists themselves.

 It is presented in many forms, including books, film and television documentaries, magazine articles, and web pages.
</item 6>

<item 7>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 7>

<item 8>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 8>

<item 9>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 9>

<item 10>
Astrobiology Magazine
Astrobiology Magazine (exploring the solar system and beyond), or Astrobiology Mag, is an American NASA-sponsored international online popular science magazine
containing popular science content, which refers to articles for the general reader on science and technology subjects.

 The magazine reports on missions of NASA and other space agencies, as well as presents news of relevant research conducted by various institutions,
universities, and non-profit groups.

 In addition, the magazine provides a forum through which researchers and the general public can oversee the progress made in fields of study that are
associated with the science of astrobiology.

 According to Phys.org, the magazine has a "vast archive of stories covering a broad array of topics ... [and] covers science and nature topics relevant to
space, innovation and biology, with an emphasis on the existence, detection and exploration of life in the universe".

 The magazine was created by Helen Matsos, presently Chief Editor and Executive Producer, and began publication in 2000.
</item 10>
=== END ITEM id=f6314516b3f08974 ===

=== ITEM id=8177ffeffa812bfa ===
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
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 1>

<item 2>
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
</item 2>

<item 3>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 3>

<item 4>
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
</item 4>

<item 5>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 5>

<item 6>
World War II caused an estimated 50–80+ million deaths: 50–55 million civilians and 21–25 million military personnel. Major Pacific battles included Bataan
(1942; mass surrender and Death March), Attu (1943), Saipan (1944), Manila (1945; over 100,000 civilians killed), Guam (1941 defeat, 1944 recapture), and
Okinawa (1945; 82 days). Other conflicts covered include
</item 6>

<item 7>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 7>

<item 8>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 8>

<item 9>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 9>

<item 10>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 10>
=== END ITEM id=8177ffeffa812bfa ===

=== ITEM id=fbf43eaaea2e7efa ===
Gold sentences and facts:
Sentence 0: Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
  Fact 0.0: Ex parte Young is a United States Supreme Court case.
  Fact 0.1: Ex parte Young is cited as 209 U.S. 123.
  Fact 0.2: Ex parte Young is dated 1908.
  Fact 0.3: Ex parte Young allows suits in federal courts against officials acting on behalf of states of the union.
  Fact 0.4: Ex parte Young allows such suits to proceed despite the state's sovereign immunity.
  Fact 0.5: Ex parte Young allows such suits when the state acted unconstitutionally.
Sentence 1:  The case was a test of the separation of church and state with respect to education.
  Fact 1.0: The Ex parte Young case was a test of the separation of church and state.
  Fact 1.1: The test of church and state separation in the Ex parte Young case concerned education.

Retrieved items:
<item 1>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 1>

<item 2>
Ex parte Young
Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of
the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
</item 2>

<item 3>
Bethel School District v. Fraser
Bethel School District v. Fraser, 478 U.S. 675 (1986), was a landmark decision by the United States Supreme Court involving free speech in public schools.

 High school student Matthew Fraser was suspended from school in the Bethel School District for making a speech including sexual double entendres at a school
assembly.

 The Supreme Court held that his suspension did not violate the First Amendment.
</item 3>

<item 4>
Seminole Tribe of Florida v. Florida
Seminole Tribe of Florida v. Florida, 517 U.S. 44 (1996) , was a United States Supreme Court case which held that Article One of the U.S. Constitution did not
give the United States Congress the power to abrogate the sovereign immunity of the states that is further protected under the Eleventh Amendment.

 Such abrogation is permitted where it is necessary to enforce the rights of citizens guaranteed under the Fourteenth Amendment as per "Fitzpatrick v. Bitzer"
427 U.S. 445 (1976).

 The case also held that the doctrine of "Ex parte Young", 209 U.S. 123 (1908), which allows state officials to be sued in their official capacity for
prospective injunctive relief, was inapplicable under these circumstances, because any remedy was limited to the one that Congress had provided.
</item 4>

<item 5>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 5>

<item 6>
Criminal law in the Waite Court
Notable cases include "United States v. Cruikshank" (1875), "United States v. Reese" (1875), "Reynolds v. United States" (1878), "Wilkerson v. Utah" (1879), the
"Trade-Mark Cases" (1879), "Strauder v. West Virginia" (1880), "Pace v. Alabama" (1883), "United States v. Harris" (1883), "Ex parte Crow Dog" (1883), "Hurtado
v. California" (1884), "Clawson v. United States" (1885), "Yick Wo v. Hopkins" (1886), "United States v. Kagama" (1886), "Ker v. Illinois" (1886), and "Mugler
v. Kansas" (1887).
</item 6>

<item 7>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 7>

<item 8>
Criminal law in the Chase Court
The Chase Court (1864–1873) issued thirty-five opinions in criminal cases over nine years, at a significantly higher rate than the Marshall Court or Taney Court
before it.

 Notable such cases include "Ex parte Milligan" (1866), "Pervear v. Massachusetts" (1866), "Ex parte McCardle" (1867, 1869), "Ex parte Yerger" (1868), and
"United States v. Kirby" (1868).
</item 8>

<item 9>
Duncan v. Kahanamoku
Duncan v. Kahanamoku, 327 U.S. 304 (1946), was a decision by the United States Supreme Court.

 It is often associated with the Japanese exclusion cases ("Hirabayashi v. United States", "Korematsu v. United States" and "Ex parte Endo") because it involved
wartime curtailment of fundamental civil liberties under the aegis of military authority, though in this case neither the plaintiff nor the nominal defendant
were Japanese.
</item 9>

<item 10>
Dannel McCollum
Dannel McCollum was the mayor of Champaign, Illinois from 1987-1999.

 He is the author of the book "The Lord Was Not on Trial" about his mother Vashti McCollum's landmark 1948 Supreme Court case McCollum v. Board of Education,
which struck down religious education in public schools.

 He also wrote the book "Remembering Champaign County".
</item 10>
=== END ITEM id=fbf43eaaea2e7efa ===

=== ITEM id=7ca87b5684cf0bb0 ===
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
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 1>

<item 2>
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 2>

<item 3>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 3>

<item 4>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 4>

<item 5>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 5>

<item 6>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 6>

<item 7>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 7>

<item 8>
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
</item 8>

<item 9>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 9>

<item 10>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 10>
=== END ITEM id=7ca87b5684cf0bb0 ===

=== ITEM id=e8de9f59078b5f56 ===
Gold sentences and facts:
Sentence 0: In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
  Fact 0.0: The Fader published an interview with CupcakKe in September.
  Fact 0.1: In the interview, CupcakKe discussed her new album.
  Fact 0.2: CupcakKe's new album would address "rape, and everything else".
Sentence 1: The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.
  Fact 1.0: The Fader, stylized as The FADER, is a New York City-based music magazine.
  Fact 1.1: The Fader was launched in 1999.
  Fact 1.2: The Fader was launched by Rob Stone and Jon Cohen.
  Fact 1.3: The Fader covers music, style and culture.
Sentence 2:  It was the first print publication to be released on iTunes.
  Fact 2.0: The Fader was the first print publication to be released on iTunes.

Retrieved items:
<item 1>
The entries profile publishers, periodicals, and literary figures: Random House imprints Villard and Simon & Schuster; Macedonian weekly *Makedonsko Sonce*;
Nepalese regent and pioneering author Queen Tripurasundari; print-on-demand firm Mimeo; stamp reference Minkus; radical Black British publisher
Bogle-L’Ouverture; spiritual media company Sounds True; historical reprint specialist Applewood; podsafe music network PMN; speculative-fiction zine *LC
</item 1>

<item 2>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 2>

<item 3>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 3>

<item 4>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 4>

<item 5>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 5>

<item 6>
Audacious (CupcakKe album)
In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
</item 6>

<item 7>
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 7>

<item 8>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 8>

<item 9>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 9>

<item 10>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 10>
=== END ITEM id=e8de9f59078b5f56 ===

=== ITEM id=7810de38138b8c42 ===
Gold sentences and facts:
Sentence 0: Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.
  Fact 0.0: Nelson Algren was born March 28, 1909.
  Fact 0.1: Nelson Algren died May 9, 1981.
  Fact 0.2: Nelson Algren was an American writer.
Sentence 1: Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
  Fact 1.0: Nathanael West was born Nathan Weinstein.
  Fact 1.1: Nathanael West was born October 17, 1903.
  Fact 1.2: Nathanael West died December 22, 1940.
  Fact 1.3: Nathanael West was an American author.
  Fact 1.4: Nathanael West was an American screenwriter.
  Fact 1.5: Nathanael West was an American satirist.

Retrieved items:
<item 1>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 1>

<item 2>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 2>

<item 3>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 3>

<item 4>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 4>

<item 5>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 5>

<item 6>
The Man with the Golden Arm (novel)
The Man with the Golden Arm is a novel by Nelson Algren, published by Doubleday in November 1949.

 One of the seminal novels of post-World War II American letters, "The Man with the Golden Arm" is widely considered Algren's greatest and most enduring work.

 It won the National Book Award in 1950.
</item 6>

<item 7>
Nathanael West
Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
</item 7>

<item 8>
A Walk on the Wild Side
A Walk on the Wild Side is a 1956 novel by Nelson Algren, most often quoted as the source for Algren's "three rules of life": "Never play cards with a man
called Doc.

 Never eat at a place called Mom's.

 Never sleep with a woman whose troubles are worse than your own."
</item 8>

<item 9>
Nelson Algren (1909–1981), an American writer, published *The Man with the Golden Arm* in 1949; it won the 1950 National Book Award and was adapted into a 1955
film. His 1956 novel *A Walk on the Wild Side* is known for his three rules of life: avoid cards with “Doc,” food at “Mom’s,” and partners with worse troubles
than yours.

Nelson DeMille (born 1943
</item 9>

<item 10>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 10>
=== END ITEM id=7810de38138b8c42 ===

=== ITEM id=f3afd6f16d4736f8 ===
Gold sentences and facts:
Sentence 0: Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.
  Fact 0.0: Migene González-Wippler is a Puerto Rican.
  Fact 0.1: Migene González-Wippler is a new-age author.
  Fact 0.2: Migene González-Wippler is a leading expert on the Afro-Caribbean religion of Santería.
Sentence 1: Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among West African descendants.
  Fact 1.0: Santería is also known as Regla de Ochá.
  Fact 1.1: Santería is also known as La Regla de Ifá.
  Fact 1.2: Santería is also known as Lucumi.
  Fact 1.3: Santería is an Afro-American religion.
  Fact 1.4: Santería is a religion of Caribbean origin.
  Fact 1.5: Santería developed in the Spanish Empire.
  Fact 1.6: Santería developed among West African descendants.

Retrieved items:
<item 1>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 1>

<item 2>
The entries describe Hawaiian muumuus, formal floral or subdued dresses also popular for maternity wear; Santería, an Afro-Caribbean Yoruba-influenced religion
syncretized with Catholicism; Scanian spit-cake and dialect; and Vanniyar origins and traditional military role. They survey folklore and traditions worldwide:
Scandinavian spirits, selkies, Pukwudgies, Hawaiian Nightmarchers, Okinawan shisa, Taiwanese Ba Jia Jiang, Bulgarian
</item 2>

<item 3>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 3>

<item 4>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 4>

<item 5>
Ewe people
The Ewe people (Ewe: "Eʋeawó" , lit.

 "Ewe people"; or " Eʋedukɔ́ ", lit.

 "Ewe nation","Eʋenyigba" Eweland;) are an African ethnic group.

 They are the largest ethnic group in Togo (32%), the third largest ethnic group in Ghana (14%), and are a minority ethnic group in southern Benin, Ivory Coast
and Nigeria.

 They speak the Ewe language (Ewe: "Eʋegbe" ) which belongs to the Niger-Congo family of languages.

 They are related to other speakers of Gbe languages, such as, the Fon, Gen, Phla Phera, and the Aja people of Togo and Benin.
</item 5>

<item 6>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 6>

<item 7>
Migene González-Wippler
Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.

 González-Wippler was born in Puerto Rico and has degrees in psychology and anthropology from the University of Puerto Rico and from Columbia University.

 In addition to her solid background in social sciences she has also worked as a science editor for the Interscience Division of John Wiley, the American
Institute of Physics, and the American Museum of Natural History, and as an English editor for the United Nations in Vienna, where she resided for many years.

 She is a cultural anthropologist and lectures frequently at universities and other educational institutions.

 She also has contributed extensively to the Institute of Puerto Rican Culture's collection, especially when it comes to Santeria and its practices, beliefs and
organization.
</item 7>

<item 8>
Awithlaknannai Mosona
It is thought that the Spanish had brought Alquerque to the American Southwest, and Awithlaknannai Mosona may have been an evolution from Alquerque.

 However, in Stewart Culin's 1907 book, the Zunis claim that they had adopted a hunt game from Mexico similar to Catch the Hare and the Fox games of Europe, and
transformed it into Awithlaknannai Mosona.

 In these games, one player has more pieces over the other, however, the other player's piece has more powers.

 The Zuni's equalized the numbers of pieces and their powers, and also may have transformed the board making its length far exceed its width.

 Diagonal lines also replaced orthogonal lines altogether.

 However, the hunt game from Mexico may have used an Alquerque board even though the game mechanics of their new game, Awithlaknannai Mosona, were completely
different.
</item 8>

<item 9>
Ethnic religion
In religious studies, ethnic religion (or indigenous religion) is the expression of religion associated with a particular ethnic group.

 Ethnic religions are often distinguished from religions which claim to not be limited in ethnic or national scope, such as Christianity or Islam.

 Ethnic religions do not have to be excluded to independent religions.

 Some localized denominations of global religions are practiced solely by certain ethnic groups, with ethnic groups like the Assyrians following a unique
denominational structure of Christianity known as the Assyrian Church of the East.
</item 9>

<item 10>
Mactan, a densely populated Cebu island, has 470,000 residents. Sino-Tibetan is a 400-plus-language family. Sikh culture intertwines faith and Punjabi heritage.
Amka is an Israeli moshav founded by Yemenite Jews. Endangered Costa Rican Maléku and Cabécar contrast with widely spoken Paraguayan Guarani.
Indigenous-language initiatives include Australia’s Ngapartji Ngapartji project. Haisla live in British
</item 10>
=== END ITEM id=f3afd6f16d4736f8 ===

=== ITEM id=0b99af58766d3356 ===
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
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 1>

<item 2>
Fallow deer
The fallow deer ("Dama dama") is a ruminant mammal belonging to the family Cervidae.

 This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar,
Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the
United States, the Falkland Islands, and Peru.

 Some taxonomers include the rarer Persian fallow deer as a subspecies ("D. d. mesopotamica"), while others treat it as an entirely different species ("D.
mesopotamica").
</item 2>

<item 3>
Forests of Mara and Mondrem
The Forests of Mara and Mondrem were adjacent medieval forests in Cheshire, England, which in the 11th century extended to over 60 sqmi , stretching from the
Mersey in the north almost to Nantwich in the south, and from the Gowy in the west to the Weaver in the east.

 Mara and Mondrem were a hunting forest of the Norman Earls of Chester, established soon after 1071 by the first earl, Hugh d'Avranches.

 They might earlier have been an Anglo-Saxon hunting forest.

 Game included wild boar, and red, fallow and roe deer.
</item 3>

<item 4>
Visayan spotted deer
The Visayan spotted deer ("Rusa alfredi"), also known as the Philippine spotted deer, is a nocturnal and endangered species of deer located primarily in the
rainforests of the Visayan islands of Panay and Negros though it once roamed other islands such as Cebu, Guimaras, Leyte, Masbate, and Samar.

 It is one of three endemic deer species in the Philippines, although it was not recognized as a separate species until 1983.

 An estimated 2,500 mature individuals survived worldwide as of 1996, according to the IUCN, although it is uncertain of how many of them still survive in the
wild.

 The diet of the deer, which consists of a variety of different types of grasses, leaves, and buds within the forest, is the primary indicator of its habitat.

 Since 1991 the range of the species has severely decreased and is now almost co-extensive with that of the Visayan warty pig.
</item 4>

<item 5>
Indian muntjac
The Indian muntjac ("Muntiacus muntjak"), also called red muntjac and barking deer, is a common muntjac deer species in South and Southeast Asia.

 It is listed as Least Concern on the IUCN Red List.
</item 5>

<item 6>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 6>

<item 7>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 7>

<item 8>
Red deer
The red deer ("Cervus elaphus") is one of the largest deer species.

 The red deer inhabits most of Europe, the Caucasus Mountains region, Asia Minor, Iran, parts of western Asia, and central Asia.

 It also inhabits the Atlas Mountains region between Morocco and Tunisia in northwestern Africa, being the only species of deer to inhabit Africa.

 Red deer have been introduced to other areas, including Australia, New Zealand, United States, Canada, Peru, Uruguay, Chile and Argentina.

 In many parts of the world, the meat (venison) from red deer is used as a food source.
</item 8>

<item 9>
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
</item 9>

<item 10>
Sri Lankan sambar deer
The Sri Lankan sambar deer ("Rusa unicolor unicolor") is a subspecies of sambar deer that lives in Sri Lanka.

 This subspecies is one of the largest sambar deer species with the largest antlers both in size and in body proportions.

 Large males weight up to 270–280 kg.

 Sri Lankan sambar live in lowland dry forests and mountain forests.

 Large herds of sambar deer roam the Horton Plains National Park, where it is the most common large mammal.
</item 10>
=== END ITEM id=0b99af58766d3356 ===

=== ITEM id=eaf10299234e717b ===
Gold sentences and facts:
Sentence 0: Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.
  Fact 0.0: Hou Hsiao-hsien was born on 8 April 1947.
  Fact 0.1: Hou Hsiao-hsien is a Taiwanese film director.
  Fact 0.2: Hou Hsiao-hsien is a Taiwanese screenwriter.
  Fact 0.3: Hou Hsiao-hsien is a Taiwanese producer.
  Fact 0.4: Hou Hsiao-hsien is a Taiwanese actor.
Sentence 1:  He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.
  Fact 1.0: Hou Hsiao-hsien is a leading figure in world cinema.
  Fact 1.1: Hou Hsiao-hsien is a leading figure in Taiwan's New Wave cinema movement.
  Fact 1.2: Hou Hsiao-hsien is an auteur.
Sentence 2:  He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in 2015 for "The Assassin" (2015).
  Fact 2.0: Hou Hsiao-hsien won the Golden Lion at the Venice Film Festival in 1989.
  Fact 2.1: Hou Hsiao-hsien won the Golden Lion for his film "A City of Sadness" (1989).
  Fact 2.2: Hou Hsiao-hsien won the Best Director award at the Cannes Film Festival in 2015.
  Fact 2.3: Hou Hsiao-hsien won the Best Director award at Cannes for his film "The Assassin" (2015).
Sentence 3:  Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
  Fact 3.0: "The Puppetmaster" (1993) is a highly regarded work by Hou Hsiao-hsien.
  Fact 3.1: "Flowers of Shanghai" (1998) is a highly regarded work by Hou Hsiao-hsien.
Sentence 4: Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.
  Fact 4.0: Kenny Ortega (Kenneth John Ortega) was born on April 18, 1950.
  Fact 4.1: Kenny Ortega is an American producer.
  Fact 4.2: Kenny Ortega is an American director.
  Fact 4.3: Kenny Ortega is an American choreographer.
Sentence 5:  He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
  Fact 5.0: Kenny Ortega is best known for directing "Hocus Pocus".
  Fact 5.1: Kenny Ortega is best known for directing the "High School Musical" trilogy.
  Fact 5.2: Kenny Ortega is best known for directing "Descendants".
  Fact 5.3: Kenny Ortega is best known for directing Michael Jackson's "This Is It" concerts.

Retrieved items:
<item 1>
Graph query row:
person: Hou Hsiao-hsien
award: None
relationship: None
years: None
year: None
event: None
</item 1>

<item 2>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 2>

<item 3>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 3>

<item 4>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 4>

<item 5>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 5>

<item 6>
Graph query row:
person: Kenny Ortega
award: None
relationship: None
years: None
year: None
event: None
</item 6>

<item 7>
The Assassin (2015 film)
The Assassin (; or: "The Assassin Niè Yǐnniáng") is a 2015 wuxia film directed by Taiwanese director Hou Hsiao-hsien.

 A Taiwan/China/Hong Kong co-production, it was an official selection in the main competition section at the 2015 Cannes Film Festival.

 At Cannes, Hou won the award for Best Director.

 It was released in China and Hong Kong on 27 August, and a day later in Taiwan on 28 August 2015.

 It was selected as the Taiwanese entry for the Best Foreign Language Film at the 88th Academy Awards but it was not nominated.
</item 7>

<item 8>
Hou Hsiao-hsien
Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.

 He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.

 He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in
2015 for "The Assassin" (2015).

 Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
</item 8>

<item 9>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 9>

<item 10>
Kenny Ortega
Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.

 He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
</item 10>
=== END ITEM id=eaf10299234e717b ===

=== ITEM id=e29f303419ce1365 ===
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
Garcia Plays Dylan
Garcia Plays Dylan is an album composed of various live performances featuring Jerry Garcia playing covers of Bob Dylan songs.

 It is culled from performances from 1973–1995, and features Garcia playing with Grateful Dead, Legion of Mary, Jerry Garcia Band, and Garcia-Saunders.

 Garcia takes lead vocals on all tracks.
</item 4>

<item 5>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 5>

<item 6>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 6>

<item 7>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 7>

<item 8>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 8>

<item 9>
Sugaree
"Sugaree" is a song written by long-time Grateful Dead lyricist Robert Hunter and composed by guitarist Jerry Garcia.

 It was written for Jerry Garcia's first solo album "Garcia", which was released in January 1972.

 As with the songs on the rest of the album, Garcia plays every instrument himself (except drums, played by Bill Kreutzmann), including acoustic guitar, bass
guitar, and an electric guitar played through a Leslie speaker.
</item 9>

<item 10>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 10>
=== END ITEM id=e29f303419ce1365 ===

=== ITEM id=2b4032c3ebf86893 ===
Gold sentences and facts:
Sentence 0:  The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
  Fact 0.0: The ground was home to the Boston Reds.
  Fact 0.1: The Boston Reds played in the Players' League in 1890.
  Fact 0.2: The Boston Reds played in the American Association in 1891.
Sentence 1:  It is home to the Port of Boston, a major shipping facility in the northeastern United States.
  Fact 1.0: The ground is home to the Port of Boston.
  Fact 1.1: The Port of Boston is a major shipping facility.
  Fact 1.2: The Port of Boston is in the northeastern United States.

Retrieved items:
<item 1>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 1>

<item 2>
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
</item 2>

<item 3>
New Bedford Bay Sox
The New Bedford Bay Sox are a baseball team that plays in the New England Collegiate Baseball League, a collegiate summer baseball league located in the
northeastern United States region of New England.

 The team is located in New Bedford, Massachusetts.

 New Bedford's Paul Walsh Field serves as the home field of the Bay Sox.

 Th NECBL's players use wooden bats, as opposed to aluminum, and are college baseball players recruited from the top intercollegiate baseball programs in the
United States.

 The franchise has appeared in league finals four times in their history, all as the Torrington Twisters (1997, 1998, 2003, 2006), and have twice played host to
the NECBL All-Star Game as the Twisters (1998, 2008).

 Pat and Beth O'Connor, owners of Little Fenway and Little Wrigley joined the ownership team of the New Bedford Bay Sox in October, 2011, and signed a 3-year
lease to play at Paul Walsh Field.

 In November 2011, former MLB player Rick Miller was announced as the 2012 Bay Sox manager.
</item 3>

<item 4>
New Bedford Bay Sox
Miller was succeeded as manager prior to the 2015 season by Westport, MA native and former Boston Red Sox draft pick Kyle Fernandes.

 Fernandes has led the Bay Sox to back-to-back postseason appearances in his two seasons as manager, most recently losing in a three-game series to the eventual
NECBL Champions the Mystic Schooners.
</item 4>

<item 5>
Braves Field
Braves Field was a baseball park in the Northeastern United States, located in Boston, Massachusetts.

 Today the site is home to Nickerson Field on the campus of Boston University.

 The stadium was home of the Boston Braves of the National League from 1915–1952, prior to the Braves' move to Milwaukee in 1953.

 The stadium hosted the 1936 Major League Baseball All-Star Game and Braves home games during the 1948 World Series.

 The Boston Red Sox used Braves Field for their home games in the 1915 and 1916 World Series since the stadium had a larger seating capacity than Fenway Park.

 Braves Field was the site of Babe Ruth's final season, playing for the Braves in 1935.

 From 1929 to 1932, the Boston Red Sox played select regular season games periodically at Braves Field.

 On May 1, 1926, Braves Field hosted the longest baseball game in history – 26 Innings.

 The game ended in a 1–1 tie.
</item 5>

<item 6>
Transportation in Boston
The Boston transportation system includes roadway, subway, regional rail, air, and sea options for passenger and freight transit in Boston, Massachusetts.

 The Massachusetts Port Authority (Massport) operates the Port of Boston, which includes a container shipping facility in South Boston, and Logan International
Airport, in East Boston.

 The Massachusetts Bay Transportation Authority (MBTA) operates bus, subway, short distance rail, and water ferry passenger services throughout the city and
region.

 Amtrak operates passenger rail service to and from major northeastern cities.

 A major bus terminal at South Station is served by varied intercity bus companies.

 The city is bisected by major highways I-90 and I-93, the intersection of which has undergone a major renovation, nicknamed the Big Dig.
</item 6>

<item 7>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 7>

<item 8>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 8>

<item 9>
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
</item 9>

<item 10>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 10>
=== END ITEM id=2b4032c3ebf86893 ===
