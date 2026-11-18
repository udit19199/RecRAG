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

=== ITEM id=8f7a5b75b04d258b ===
Gold sentences and facts:
Sentence 0: Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.
  Fact 0.0: Adam Fox lived from 1883 to 1977.
  Fact 0.1: Adam Fox was a Canon.
  Fact 0.2: Adam Fox was the Dean of Divinity at Magdalen College, Oxford.
Sentence 1:  He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.
  Fact 1.0: Adam Fox was one of the first members of the Inklings.
  Fact 1.1: The Inklings was a literary group.
  Fact 1.2: The Inklings also included C.S. Lewis.
  Fact 1.3: The Inklings also included J.R.R. Tolkien.
Sentence 2: The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and late 1949.
  Fact 2.0: The Inklings was an informal literary discussion group.
  Fact 2.1: The Inklings was associated with the University of Oxford, England.
  Fact 2.2: The Inklings lasted for nearly two decades.
  Fact 2.3: The Inklings operated between the early 1930s and late 1949.

Retrieved items:
<item 1>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 1>

<item 2>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 2>

<item 3>
The text profiles diverse educational and related institutions worldwide. Highlights include Berat University (Albania), founded in 2009 and expanding to four
faculties; Rotman (Toronto), established in 1950; Auburn, Arizona, Debrecen, Copenhagen, Dalhousie and other research universities; specialist schools such as
Falmouth and Aalborghus; and private or state colleges across Ghana, Malaysia, Norway, Pakistan and elsewhere. It also covers Syracuse sororities, a
</item 3>

<item 4>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 4>

<item 5>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 5>

<item 6>
Inklings
The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and
late 1949.

 The Inklings were literary enthusiasts who praised the value of narrative in fiction and encouraged the writing of fantasy.
</item 6>

<item 7>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 7>

<item 8>
Andrew Benjamin
Andrew Benjamin (born 1952, Australia) is an Australian philosopher.

 Currently he holds two academic posts: Distinguished Professor of Philosophy and the Humanities at Kingston University, London and Professor of Philosophy and
Jewish Thought in the Department of Philosophy and the Australian Centre for Jewish Civilization at Monash University, Melbourne, Australia.

 Benjamin first came to critical attention with his writings in continental philosophy, writing articles and editing books on the thinking of Martin Heidegger,
Theodor Adorno, Walter Benjamin, Julia Kristeva and Jean-François Lyotard.

 Benjamin has become involved in the field of architecture, to the extent that he has also taught in various schools of architecture in UK, US and Australia.
</item 8>

<item 9>
Eagle Street College
The Eagle Street College was an informal literary society established in 1885 at the home of James William Wallace in Eagle Street, Bolton, to read and discuss
literary works, particularly the poetry of Walt Whitman, (1819–91).

 The group subsequently became known as the Bolton Whitman Fellowship or Whitmanites.

 Its founder members were Wallace, Dr John Johnston and Fred Wild.

 The group held an annual 'Whitman Day' celebration around the 31 May which was the poet's birthday.
</item 9>

<item 10>
The entries profile scholars, writers, artists, and literary groups. Wale Adebanwi became Oxford’s first Black African Rhodes Professor. Naturalist Witmer Stone
led major ornithological organizations, worked at Philadelphia’s Academy of Natural Sciences, and wrote landmark works on New Jersey plants and Cape May birds.
Oxford’s Inklings included Colin Hardie, Charles Leslie Wrenn, Adam Fox, and Roger Lancelyn Green; Bolton’s Eagle Street College celebrated Walt Whitman. Other
figures include R
</item 10>
=== END ITEM id=8f7a5b75b04d258b ===

=== ITEM id=6c42632bf4460755 ===
Gold sentences and facts:
Sentence 0: Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.
  Fact 0.0: Pietro Alessandro Gaspare Scarlatti was born on 2 May 1660.
  Fact 0.1: Pietro Alessandro Gaspare Scarlatti died on 22 October 1725.
  Fact 0.2: Pietro Alessandro Gaspare Scarlatti was an Italian Baroque composer.
  Fact 0.3: Pietro Alessandro Gaspare Scarlatti was especially famous for his operas.
  Fact 0.4: Pietro Alessandro Gaspare Scarlatti was especially famous for his chamber cantatas.
Sentence 1: Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky, was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.
  Fact 1.0: The Russian name of Pyotr Ilyich Tchaikovsky is Пётр Ильи́ч Чайко́вский.
  Fact 1.1: Pyotr Ilyich Tchaikovsky was born on 25 April/7 May 1840.
  Fact 1.2: Pyotr Ilyich Tchaikovsky died on 25 October/6 November 1893.
  Fact 1.3: Pyotr Ilyich Tchaikovsky is often anglicized as Peter Ilich Tchaikovsky.
  Fact 1.4: Pyotr Ilyich Tchaikovsky was a Russian composer of the romantic period.
  Fact 1.5: Some works by Pyotr Ilyich Tchaikovsky are among the most popular music in the classical repertoire.
Sentence 2:  He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the United States.
  Fact 2.0: Tchaikovsky was the first Russian composer whose music made a lasting impression internationally.
  Fact 2.1: The international impression of Tchaikovsky's music was bolstered by his appearances as a guest conductor.
  Fact 2.2: Tchaikovsky appeared as a guest conductor in Europe.
  Fact 2.3: Tchaikovsky appeared as a guest conductor in the United States.

Retrieved items:
<item 1>
Tchaikovsky (film)
Tchaikovsky (Russian: Чайковский ) is a 1970 Soviet biopic film directed by Igor Talankin.

 It featured Innokenty Smoktunovsky in the role of the famous Russian composer Pyotr Ilyich Tchaikovsky.

 It was nominated for the 1971 Academy Award for Best Foreign Language Film as well as the Academy Award for Original Song Score and Adaptation.
</item 1>

<item 2>
Symphony No. 2 (Rachmaninoff)
Symphony No. 2 in E minor, Op. 27 is a symphony by the Russian composer Sergei Rachmaninoff, written in 1906–07.

 The premiere was conducted by the composer himself in Saint Petersburg on 8 February 1908.

 Its duration is approximately 60 minutes when performed uncut; cut performances can be as short as 35 minutes.

 The score is dedicated to Sergei Taneyev, a Russian composer, teacher, theorist, author, and pupil of Pyotr Ilyich Tchaikovsky.

 Alongside his Piano Concerto No. 2 and Piano Concerto No. 3, this symphony remains one of the composer's best known compositions.
</item 2>

<item 3>
Piano Concerto No. 2 (Tchaikovsky)
Pyotr Ilyich Tchaikovsky's Piano Concerto No. 2 in G major, Op.

 44, was written in 1879–1880.

 It was dedicated to Nikolai Rubinstein, who had insisted he be allowed to perform it at the premiere as a way of making up for his harsh criticism of
Tchaikovsky's First Piano Concerto.

 Rubinstein was destined never to play it, however, as he died in March 1881.

 The premiere performance took place in New York City, on 12 November 1881.

 The soloist was Madeline Schiller, and Theodore Thomas conducted the New York Philharmonic orchestra.

 The first Russian performance was in Moscow in May 1882, conducted by Anton Rubinstein with Tchaikovsky's pupil, Sergei Taneyev, at the piano.
</item 3>

<item 4>
Chaikovskij (crater)
Chaikovskij (sometimes Tchaikovsky) is a crater on Mercury.

 It has a diameter of 165 kilometers.

 Its name was adopted by the International Astronomical Union in 1976.

 Chaikovskij is named for the Russian composer Pyotr Ilyich Tchaikovsky, who lived from 1840 to 1893.
</item 4>

<item 5>
Pyotr Ilyich Tchaikovsky
Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky,
was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.

 He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the
United States.

 Tchaikovsky was honored in 1884, by Emperor Alexander III, and awarded a lifetime pension.
</item 5>

<item 6>
Kamarinskaya
Kamarinskaya (Russian: камаринская ) is a Russian traditional folk dance, which is mostly known today as the Russian composer Mikhail Glinka's composition of
the same name.

 Glinka's "Kamarinskaya", written in 1848, was the first orchestral work based entirely on Russian folk song and to use the compositional principles of that
genre to dictate the form of the music.

 It became a touchstone for the following generation of Russian composers ranging from the Western-oriented Pyotr Ilyich Tchaikovsky to the group of
nationalists known collectively as The Five and was also lauded abroad, most notably by French composer Hector Berlioz.
</item 6>

<item 7>
International Tchaikovsky Competition
The International Tchaikovsky Competition is a classical-music competition held every four years in Moscow, Russia, for pianists, violinists, and cellists
between 16 and 32 years of age, and singers between 19 and 32 years of age.

 The competition is named after Russian composer Pyotr Ilyich Tchaikovsky and is an active member of the World Federation of International Music Competitions.
</item 7>

<item 8>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 8>

<item 9>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 9>

<item 10>
Piano Sonata in C-sharp minor (Tchaikovsky)
The Piano Sonata in C-sharp minor, Op.

 posth.

 80, was written by Russian composer Pyotr Ilyich Tchaikovsky in 1865, his last year as a student at the St Petersburg Conservatory.

 The sonata in its original form was not published in Tchaikovsky's lifetime; it was published in 1900 by P. Jurgenson, and given the posthumous opus number 80.
</item 10>
=== END ITEM id=6c42632bf4460755 ===

=== ITEM id=8f51e068f26a2457 ===
Gold sentences and facts:
Sentence 0:  The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
  Fact 0.0: The documentary explores the history of Sunset Strip.
  Fact 0.1: Sunset Strip is a mile and a half long stretch of road.
  Fact 0.2: Sunset Strip runs through West Hollywood.
Sentence 1:  The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.
  Fact 1.0: The film features some early pioneers of the hardcore punk music scene.
  Fact 1.1: Bad Brains is an early hardcore punk pioneer featured in the film.
  Fact 1.2: Black Flag is an early hardcore punk pioneer featured in the film.
  Fact 1.3: D.O.A. is an early hardcore punk pioneer featured in the film.
  Fact 1.4: Minor Threat is an early hardcore punk pioneer featured in the film.
  Fact 1.5: Minutemen is an early hardcore punk pioneer featured in the film.
  Fact 1.6: SSD is an early hardcore punk pioneer featured in the film.
  Fact 1.7: The film features other early hardcore punk pioneers besides those named.

Retrieved items:
<item 1>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 1>

<item 2>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 2>

<item 3>
Sunset Strip
Sunset Strip is the mile-and-a-half (2.4 km) stretch of Sunset Boulevard that passes through West Hollywood, California.

 It extends from West Hollywood's eastern border with Hollywood at Crescent Heights Boulevard to its western border with Beverly Hills at Sierra Drive.

 Sunset Strip is probably the best-known portion of Sunset Boulevard, with boutiques, restaurants, rock clubs, and nightclubs.

 It is also known for its array of huge, colorful billboards.
</item 3>

<item 4>
Mayor of the Sunset Strip
Mayor of the Sunset Strip is a 2003 documentary film on the life of Rodney Bingenheimer directed by George Hickenlooper, and produced by Chris Carter.

 In 2011, Craig Hlavaty of the "Houston Press" named "Mayor of the Sunset Strip" at number eight on the paper's list of "The 31 Best Music Documentaries of All
Time".

 The film won the Best Documentary Feature at the 2004 Santa Barbara International Film Festival.
</item 4>

<item 5>
Sunset Strip curfew riots
The Sunset Strip curfew riots, also known as the "hippie riots", were a series of early counterculture-era clashes that took place between police and young
people on the Sunset Strip in Hollywood, California, beginning in the mid-1966 and continuing on and off through the early 1970s.
</item 5>

<item 6>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 6>

<item 7>
Hellfest (American music festival)
Hellfest was an American hardcore punk festival.

 Named for the intense summer heat at the show, the festival originally began featuring hardcore bands before crossing into several genres later on.

 Hellfest was known for its fan-friendly atmosphere, (including a lack of barriers, providing easy access to the stage for stage diving) and for its high
social-awareness, focusing on such causes as animal rights, with many attendees following the straight edge and/or vegan lifestyle.

 It continued as an annual event until it was cancelled in 2005.

 After its cancellation less than thirty-six hours before doors were scheduled to open on the festival's ninth year, some fans, unhappy with the inadequate
processing of refunds, filed small claims court lawsuits against promoter Keith Allen.
</item 7>

<item 8>
Sunset Strip (2012 film)
Sunset Strip is a 2012 documentary directed by Hans Fjellestad, and produced by Tommy Alastra.

 The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
</item 8>

<item 9>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 9>

<item 10>
Riot on Sunset Strip
Riot on Sunset Strip is a 1967 counterculture-era exploitation movie, released by American International Pictures.

 It was filmed and released within four months of the late-1966 Sunset Strip curfew riots.
</item 10>
=== END ITEM id=8f51e068f26a2457 ===

=== ITEM id=d3822756e481b092 ===
Gold sentences and facts:
Sentence 0:  The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova Planta Decree of Majorca and Ibiza (28 November 1715).
  Fact 0.0: The body that was to become the Gran i General Consell was founded in 1249.
  Fact 0.1: The Consell was abolished on 22 July 1718.
  Fact 0.2: The Consell was abolished by Philip V.
  Fact 0.3: Philip V was empowered by the Nova Planta Decree of Majorca and Ibiza.
  Fact 0.4: The Nova Planta Decree of Majorca and Ibiza is dated 28 November 1715.
Sentence 1: Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January 1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
  Fact 1.0: Philip V's Spanish name is Felipe V.
  Fact 1.1: Philip V's French name is Philippe.
  Fact 1.2: Philip V's Italian name is Filippo.
  Fact 1.3: Philip V was born on 19 December 1683.
  Fact 1.4: Philip V died on 9 July 1746.
  Fact 1.5: Philip V was King of Spain from 1 November 1700 to 15 January 1724.
  Fact 1.6: Philip V abdicated in favour of his son Louis on 15 January 1724.
  Fact 1.7: Philip V reassumed the Spanish throne on 6 September 1724, upon his son's death.
  Fact 1.8: Philip V was King of Spain from 6 September 1724 until his death on 9 July 1746.

Retrieved items:
<item 1>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 1>

<item 2>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 2>

<item 3>
Papal conclave, September 1503
The papal conclave of September 1503 elected Pope Pius III to succeed Pope Alexander VI.

 Due to the Italian Wars, the College of Cardinals was surrounded by three potentially hostile armies, loyal to Louis XII of France, Ferdinand II of Aragon, and
Cesare Borgia (the cardinal-nephew of Alexander VI).
</item 3>

<item 4>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 4>

<item 5>
Pier Luigi de Borgia, 1st duke of Gandía
Pier Luigi de Borgia, 1st duke of Gandía (Spanish: "Pedro Luis de Borja" , Latin: "Petrus Ludovicus de Boria" ) (1458 or 1460–1488 or 1491) was a Valencian
noble.

 Pier Luigi was the son of Cardinal Rodrigo Borgia (later Pope Alexander VI), and half-brother of Cesare Borgia, Gioffre Borgia, Giovanni Borgia and Lucrezia
Borgia.
</item 5>

<item 6>
Philip V of Spain
Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January
1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
</item 6>

<item 7>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 7>

<item 8>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 8>

<item 9>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 9>

<item 10>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 10>
=== END ITEM id=d3822756e481b092 ===

=== ITEM id=64326fb2b79ae889 ===
Gold sentences and facts:
Sentence 0: Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.
  Fact 0.0: Sir Nicholas Robert Hytner was born on 7 May 1956.
  Fact 0.1: Sir Nicholas Robert Hytner is an English theatre director.
  Fact 0.2: Sir Nicholas Robert Hytner is an English film director.
  Fact 0.3: Sir Nicholas Robert Hytner is an English film producer.
Sentence 1: Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
  Fact 1.0: Maya Deren was born on April 29, 1917.
  Fact 1.1: Maya Deren died on October 13, 1961.
  Fact 1.2: Maya Deren was born Eleanora Derenkowskaia.
  Fact 1.3: The birth name Eleanora Derenkowskaia is given in Russian as Элеоно́ра Деренко́вская.
  Fact 1.4: Maya Deren was a Russian-American filmmaker.
  Fact 1.5: Maya Deren was one of the most important American experimental filmmakers in the 1940s and 1950s.
  Fact 1.6: Maya Deren was one of the most important American entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
Sentence 2:  Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
  Fact 2.0: Maya Deren was also a choreographer.
  Fact 2.1: Maya Deren was also a dancer.
  Fact 2.2: Maya Deren was also a film theorist.
  Fact 2.3: Maya Deren was also a poet.
  Fact 2.4: Maya Deren was also a lecturer.
  Fact 2.5: Maya Deren was also a writer.
  Fact 2.6: Maya Deren was also a photographer.

Retrieved items:
<item 1>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 1>

<item 2>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 2>

<item 3>
Filmworks X: In the Mirror of Maya Deren
Filmworks X: In the Mirror of Maya Deren features a score for film by John Zorn.

 The album was released on Zorn's own label, Tzadik Records, in 2001 and contains music that Zorn wrote and recorded for the documentary film "In the Mirror of
Maya Deren" on the life and work of Maya Deren directed by Martina Kudlácek.
</item 3>

<item 4>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 4>

<item 5>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 5>

<item 6>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 6>

<item 7>
Maya Deren
Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the
most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.

 Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
</item 7>

<item 8>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 8>

<item 9>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 9>

<item 10>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 10>
=== END ITEM id=64326fb2b79ae889 ===

=== ITEM id=a69d9158ea89719b ===
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
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 1>

<item 2>
Codename: Kids Next Door
Codename: Kids Next Door, commonly abbreviated to Kids Next Door or KND, is an American animated television series created by Tom Warburton for Cartoon Network,
and the 13th of the network's "Cartoon Cartoons".

 The series centers on the adventures of five children who operate from a high-tech tree house, fighting against adult and teen villains with advanced 2×4
technology.

 Using their codenames (Numbuh 1, Numbuh 2, Numbuh 3, Numbuh 4, and Numbuh 5), they are Sector V, part of a global organization called the Kids Next Door.
</item 2>

<item 3>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 3>

<item 4>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 4>

<item 5>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 5>

<item 6>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 6>

<item 7>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 7>

<item 8>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 8>

<item 9>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 9>

<item 10>
Honey, We Shrunk Ourselves
Honey, We Shrunk Ourselves is a 1997 live-action direct-to-video sequel to "Honey, I Shrunk the Kids" and "Honey, I Blew Up the Kid".

 It is the third and final installment in the "Honey, I Shrunk the Kids" trilogy.

 The directorial debut of cinematographer Dean Cundey and released through Walt Disney Home Video, it tells the story of inventor Wayne Szalinski as he
accidentally shrinks his wife, brother, sister-in-law, and himself with his electromagnetic shrink ray.
</item 10>
=== END ITEM id=a69d9158ea89719b ===

=== ITEM id=fc8d8157f1ea1db4 ===
Gold sentences and facts:
Sentence 0:  He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".
  Fact 0.0: The person described has written a column for Fortune magazine for more than twenty years.
  Fact 0.1: The person described spent a decade at Esquire before writing for Fortune.
Sentence 1:  Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
  Fact 1.0: The described publication was founded in 1933.
  Fact 1.1: The described publication flourished during the Great Depression.
  Fact 1.2: The described publication flourished under the guidance of founders Arnold Gingrich, David A. Smart, and Henry L. Jackson.

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
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 2>

<item 3>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 3>

<item 4>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 4>

<item 5>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 5>

<item 6>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 6>

<item 7>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 7>

<item 8>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 8>

<item 9>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 9>

<item 10>
This Is Bing Crosby
This Is Bing Crosby was a fifteen-minute five times a week daytime radio program featuring Bing Crosby acting as a disc jockey.

 Minute Maid quick frozen concentrated orange juice was promoted on the shows.
</item 10>
=== END ITEM id=fc8d8157f1ea1db4 ===

=== ITEM id=27e8cd5eff5a5e83 ===
Gold sentences and facts:
Sentence 0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
  Fact 0.0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia.
  Fact 0.1: The 1997 European Athletics Junior Championships were held on July 24-27.
Sentence 1: Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.
  Fact 1.0: Ljubljana is the capital of Slovenia.
  Fact 1.1: Ljubljana is the largest city of Slovenia.
  Fact 1.2: Ljubljana is also known by other, historical names.

Retrieved items:
<item 1>
European Athletics U20 Championships
The European Athletics U20 Championships (formerly named the European Athletics Junior Championships up to 2015) are the European championships for athletes who
are 19 years of age or under, which is the age range recognised by the IAAF as junior athletes.

 The event is currently organized by the European Athletic Association.
</item 1>

<item 2>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 2>

<item 3>
1997 European Athletics Junior Championships
The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
</item 3>

<item 4>
Ljubljana
Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.

 It has been the cultural, educational, economic, political, and administrative center of independent Slovenia since 1991.

 Its central geographic location within Slovenia, transport connections, concentration of industry, scientific and research institutions, and cultural tradition
are contributing factors to its leading position.
</item 4>

<item 5>
1996 IPB Czech Indoor – Doubles
Jonas Björkman and Javier Frana was the defending champions but did not compete that year.
</item 5>

<item 6>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 6>

<item 7>
Amela Terzić
Amela Terzić (Serbian Cyrillic: Амела Терзић, born 2 January 1993 in Priboj, FR Yugoslavia) is a Serbian middle-distance runner.

 She won two gold medals at 2011 European Athletics Junior Championships in Tallinn and was the junior champion at the 2012 European Cross Country
Championships.

 She has also been a medallist in the 1500 metres at the World Junior and World Youth Championships and a gold medalist at the 2013 European U23 Championship.
</item 7>

<item 8>
1968 European Junior Games
The 1968 European Junior Games was the third edition of the biennial athletics competition for European athletes aged under twenty.

 It was the final edition of the competition under that name and was succeeded by the European Athletics Junior Championships in 1970.

 The event was held in Leipzig, part of the German Democratic Republic at that time.

 The competition was mainly between Eastern European countries, with Belgium, Austria, Netherlands and Greece being present from the west.
</item 8>

<item 9>
2015 European Athletics Junior Championships
The 2015 European Athletics Junior Championships were the 23rd edition of the biennial European U20 athletics championships.

 They were held in Eskilstuna, Sweden from 16 July to 19 July.
</item 9>

<item 10>
1964 European Junior Games
The 1964 European Junior Games was the first edition of the biennial athletics competition for European athletes aged under twenty.

 It was an unofficial competition without sanction from the European Athletic Association.

 The event was held at the 10th-Anniversary Stadium in Warsaw, Poland.

 The competition eventually led to the creation of the European Athletics Junior Championships in 1970.
</item 10>
=== END ITEM id=27e8cd5eff5a5e83 ===

=== ITEM id=72ef605937015123 ===
Gold sentences and facts:
Sentence 0:  The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.
  Fact 0.0: The first edition was written by J. Eric Holmes.
  Fact 0.1: The first edition was based on the original work of Gary Gygax and Dave Arneson.
Sentence 1: John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine, an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and science fiction.
  Fact 1.0: John Eric Holmes M.D. was born on February 16, 1930.
  Fact 1.1: John Eric Holmes M.D. died on March 20, 2010.
  Fact 1.2: John Eric Holmes M.D. was an associate professor of neurology at the University of Southern California School of Medicine.
  Fact 1.3: John Eric Holmes M.D. was an author and promoter of fantasy role-playing games.
  Fact 1.4: John Eric Holmes M.D. was a noted fan and enthusiast of Edgar Rice Burroughs.
  Fact 1.5: John Eric Holmes M.D. was an American writer of non-fiction, fantasy and science fiction.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 1>

<item 2>
Dungeons &amp; Dragons Companion Set
The Dungeons & Dragons Companion Set is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1984 as an expansion to the "Dungeons & Dragons Basic Set".
</item 2>

<item 3>
Dungeons &amp; Dragons Basic Set
The Dungeons & Dragons Basic Set is a set of rulebooks for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 First published in 1977, it saw a handful of revisions and reprintings.

 The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.

 Later editions were edited by Tom Moldvay, Dave Cook, and Frank Mentzer.
</item 3>

<item 4>
The Dungeons & Dragons Basic Set, first published in 1977 and revised by Holmes, Moldvay, Cook, and Mentzer, inspired several expansions and beginner products.
These include the Companion Set (1984), Master Rules (1985), and Immortals Rules (1986); the 1991 New Easy-to-Master Game replaced the Basic Set. Beginner
adventures included Gygax’s Keep on the Borderlands (1979) and Moldvay’s partially developed, low-level
</item 4>

<item 5>
Dungeons &amp; Dragons Master Rules
Dungeons & Dragons Master Rules is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1985 as an expansion to the "Basic Set".
</item 5>

<item 6>
The Keep on the Borderlands
The Keep on the Borderlands is a "Dungeons & Dragons" module by Gary Gygax, first printed in December 1979.

 In it, player characters are based at a keep and investigate a nearby series of caves that are filled with a variety of monsters.

 It was designed to be used with the "Dungeons & Dragons Basic Set", and was included in the 1979–1982 editions of the "Basic Set".

 It was designed for people new to "Dungeons & Dragons".
</item 6>

<item 7>
Dungeons &amp; Dragons Immortals Rules
Dungeons & Dragons Immortals Rules, written by Frank Mentzer, is a boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game first published by
TSR in 1986 as an expansion to the "Basic Set".
</item 7>

<item 8>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 8>

<item 9>
GURPS Basic Set
GURPS Basic Set is a hard-bound two volume set written by Steve Jackson, Sean M. Punch, and David L. Pulver.

 The first edition "GURPS Basic Set" was published in 1986.
</item 9>

<item 10>
Dungeons &amp; Dragons Game (1991 boxed set)
The New Easy-to-Master Dungeons & Dragons Game is an introductory set for the "Dungeons & Dragons" fantasy role-playing game, published by TSR, Inc. in 1991.

 It was a replacement for the previous "Dungeons & Dragons Basic Set", serving to introduce new players to the game, using the rule set previously established.
</item 10>
=== END ITEM id=72ef605937015123 ===

=== ITEM id=a10bf64d482aaaee ===
Gold sentences and facts:
Sentence 0:  His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".
  Fact 0.0: The unnamed subject's American superior was General Dwight D. Eisenhower.
  Fact 0.1: Dwight D. Eisenhower wrote that the unnamed subject was 'blunt, at times to the point of rudeness'.
Sentence 1: Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United States from 1953 until 1961.
  Fact 1.0: Dwight David 'Ike' Eisenhower was born on October 14, 1890.
  Fact 1.1: Dwight David 'Ike' Eisenhower died on March 28, 1969.
  Fact 1.2: Dwight David 'Ike' Eisenhower was an American politician.
  Fact 1.3: Dwight David 'Ike' Eisenhower was an Army general.
  Fact 1.4: Dwight David 'Ike' Eisenhower served as the 34th President of the United States.
  Fact 1.5: Dwight David 'Ike' Eisenhower served as President from 1953 until 1961.

Retrieved items:
<item 1>
Dwight D. Eisenhower
Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United
States from 1953 until 1961.

 During World War II, he was a five-star general in the United States Army and served as Supreme Commander of the Allied Expeditionary Forces in Europe.

 He was responsible for planning and supervising the invasion of North Africa in Operation Torch in 1942–43 and the successful invasion of France and Germany in
1944–45 from the Western Front.
</item 1>

<item 2>
Bill Clinton
William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States
from 1993 to 2001.

 Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.

 Before that, he served as Arkansas attorney general, from 1977 to 1979.

 A member of the Democratic Party, Clinton was ideologically a New Democrat and many of his policies reflected a centrist "Third Way" political philosophy.
</item 2>

<item 3>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 3>

<item 4>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 4>

<item 5>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 5>

<item 6>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 6>

<item 7>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 7>

<item 8>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 8>

<item 9>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 9>

<item 10>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 10>
=== END ITEM id=a10bf64d482aaaee ===

=== ITEM id=e870f494ae313d08 ===
Gold sentences and facts:
Sentence 0:  The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.
  Fact 0.0: The group included Alexander Bogdanov.
  Fact 0.1: The group included Maxim Gorky.
  Fact 0.2: The group included Anatoly Lunacharsky.
  Fact 0.3: The group included Mikhail Pokrovsky.
  Fact 0.4: The group included Grigory Aleksinsky.
  Fact 0.5: The group included Stanislav Volski.
  Fact 0.6: The group included Martyn Liadov.
Sentence 1: Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.
  Fact 1.0: Anatoly Vasilyevich Lunacharsky's Russian name is Анато́лий Васи́льевич Лунача́рский.
  Fact 1.1: Anatoly Vasilyevich Lunacharsky was born on 23 November 1875 (Old Style 11 November 1875).
  Fact 1.2: Anatoly Vasilyevich Lunacharsky died on 26 December 1933.
  Fact 1.3: Anatoly Vasilyevich Lunacharsky was a Russian Marxist revolutionary.
  Fact 1.4: Anatoly Vasilyevich Lunacharsky was the first Soviet People's Commissar of Education.
  Fact 1.5: Anatoly Vasilyevich Lunacharsky was responsible for culture and education as People's Commissar of Education.

Retrieved items:
<item 1>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 1>

<item 2>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 2>

<item 3>
Graph query row:
person: Anatoly Lunacharsky
relationship: MEMBER_OF
connected_to: Vpered
birth_date: 23 November [O.S. 11 November] 1875
birthplace: None
</item 3>

<item 4>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 4>

<item 5>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 5>

<item 6>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 6>

<item 7>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 7>

<item 8>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 8>

<item 9>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 9>

<item 10>
Petrashevsky Circle
The Petrashevsky Circle was a Russian literary discussion group of progressive-minded commoner-intellectuals in St. Petersburg in the 1840s.

 It was organized by Mikhail Petrashevsky, a follower of the French utopian socialist Charles Fourier.

 Among the members were writers, teachers, students, minor government officials, and army officers.

 While differing in political views, most of them were opponents of the tsarist autocracy and Russian serfdom.

 Like that of the Lyubomudry group founded earlier in the century, the purpose of the circle was to discuss Western philosophy and literature that was
officially banned by the Imperial government of Tsar Nicholas I.
</item 10>
=== END ITEM id=e870f494ae313d08 ===

=== ITEM id=ceb559a5c1cb406c ===
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
Ex parte Young
Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of
the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
</item 1>

<item 2>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 2>

<item 3>
Seminole Tribe of Florida v. Florida
Seminole Tribe of Florida v. Florida, 517 U.S. 44 (1996) , was a United States Supreme Court case which held that Article One of the U.S. Constitution did not
give the United States Congress the power to abrogate the sovereign immunity of the states that is further protected under the Eleventh Amendment.

 Such abrogation is permitted where it is necessary to enforce the rights of citizens guaranteed under the Fourteenth Amendment as per "Fitzpatrick v. Bitzer"
427 U.S. 445 (1976).

 The case also held that the doctrine of "Ex parte Young", 209 U.S. 123 (1908), which allows state officials to be sued in their official capacity for
prospective injunctive relief, was inapplicable under these circumstances, because any remedy was limited to the one that Congress had provided.
</item 3>

<item 4>
Duncan v. Kahanamoku
Duncan v. Kahanamoku, 327 U.S. 304 (1946), was a decision by the United States Supreme Court.

 It is often associated with the Japanese exclusion cases ("Hirabayashi v. United States", "Korematsu v. United States" and "Ex parte Endo") because it involved
wartime curtailment of fundamental civil liberties under the aegis of military authority, though in this case neither the plaintiff nor the nominal defendant
were Japanese.
</item 4>

<item 5>
Criminal law in the Waite Court
Notable cases include "United States v. Cruikshank" (1875), "United States v. Reese" (1875), "Reynolds v. United States" (1878), "Wilkerson v. Utah" (1879), the
"Trade-Mark Cases" (1879), "Strauder v. West Virginia" (1880), "Pace v. Alabama" (1883), "United States v. Harris" (1883), "Ex parte Crow Dog" (1883), "Hurtado
v. California" (1884), "Clawson v. United States" (1885), "Yick Wo v. Hopkins" (1886), "United States v. Kagama" (1886), "Ker v. Illinois" (1886), and "Mugler
v. Kansas" (1887).
</item 5>

<item 6>
Dannel McCollum
Dannel McCollum was the mayor of Champaign, Illinois from 1987-1999.

 He is the author of the book "The Lord Was Not on Trial" about his mother Vashti McCollum's landmark 1948 Supreme Court case McCollum v. Board of Education,
which struck down religious education in public schools.

 He also wrote the book "Remembering Champaign County".
</item 6>

<item 7>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 7>

<item 8>
Chy Lung v. Freeman
Chy Lung v. Freeman, 92 U.S. 275 (1876) was a United States Supreme Court case where the Supreme Court ruled that the power to set rules surrounding
immigration, and to manage foreign relations, rested with the United States Federal Government, rather than with the states.

 The case has been cited in other Supreme Court cases related to government authority on matters relating to immigration policy and immigration enforcement,
most recently in "Arizona v. United States" (2012).
</item 8>

<item 9>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 9>

<item 10>
Criminal law in the Chase Court
The Chase Court (1864–1873) issued thirty-five opinions in criminal cases over nine years, at a significantly higher rate than the Marshall Court or Taney Court
before it.

 Notable such cases include "Ex parte Milligan" (1866), "Pervear v. Massachusetts" (1866), "Ex parte McCardle" (1867, 1869), "Ex parte Yerger" (1868), and
"United States v. Kirby" (1868).
</item 10>
=== END ITEM id=ceb559a5c1cb406c ===
