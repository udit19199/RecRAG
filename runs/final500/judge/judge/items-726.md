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

This file has 6 items.

=== ITEM id=16b04d8afaf6b7e8 ===
Gold sentences and facts:
Sentence 0: `` Baby , Now That I 've Found You '' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh , Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' ( Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 ( From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by The Cox Family ) `` I Will '' ( John Lennon , Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album '' performed by Alison Krauss & Union Station )
  Fact 0.0: 'Baby, Now That I've Found You' was written by Tony Macaulay and John MacLeod.
  Fact 0.1: 'Baby, Now That I've Found You' runs 3 minutes 49 seconds.
  Fact 0.2: 'Baby, Now That I've Found You' is previously unreleased.
  Fact 0.3: 'Oh, Atlanta' was written by Mick Ralphs.
  Fact 0.4: 'Oh, Atlanta' runs 4 minutes 41 seconds.
  Fact 0.5: 'Oh, Atlanta' is previously unreleased.
  Fact 0.6: 'Broadway' was written by Sidney & Suzanne Cox.
  Fact 0.7: 'Broadway' runs 3 minutes 57 seconds.
  Fact 0.8: 'Broadway' is previously unreleased.
  Fact 0.9: 'Every Time You Say Goodbye' was written by John Pennell.
  Fact 0.10: 'Every Time You Say Goodbye' runs 3 minutes 13 seconds.
  Fact 0.11: 'Every Time You Say Goodbye' is from the album 'Every Time You Say Goodbye' by Alison Krauss & Union Station.
  Fact 0.12: 'Tonight I'll Be Lonely Too' was written by Sidney Cox.
  Fact 0.13: 'Tonight I'll Be Lonely Too' runs 3 minutes 25 seconds.
  Fact 0.14: 'Tonight I'll Be Lonely Too' is from the album 'I've Got That Old Feeling' by Alison Krauss.
  Fact 0.15: 'Teardrops Will Kiss the Morning Dew' was written by Paul Craft.
  Fact 0.16: 'Teardrops Will Kiss the Morning Dew' runs 3 minutes 38 seconds.
  Fact 0.17: 'Teardrops Will Kiss the Morning Dew' is from the album 'Two Highways' by Alison Krauss & Union Station.
  Fact 0.18: 'Sleep On' was written by Nelson Mandrell.
  Fact 0.19: 'Sleep On' runs 2 minutes 23 seconds.
  Fact 0.20: 'Sleep On' is from the album 'Too Late to Cry' by Alison Krauss.
  Fact 0.21: 'When God Dips His Pen of Love in My Heart' was written by Cleavant Derricks.
  Fact 0.22: 'When God Dips His Pen of Love in My Heart' runs 2 minutes 26 seconds.
  Fact 0.23: 'When God Dips His Pen of Love in My Heart' is from the album 'Everybody's Reaching Out for Someone' by The Cox Family.
  Fact 0.24: 'I Will' was written by John Lennon and Paul McCartney.
  Fact 0.25: 'I Will' runs 4 minutes 4 seconds.
  Fact 0.26: 'I Will' is from the album 'Within Reach' by Tony Furtado.
  Fact 0.27: 'I Don't Believe You've Met My Baby' was written by Autry Inman.
  Fact 0.28: 'I Don't Believe You've Met My Baby' runs 3 minutes 10 seconds.
  Fact 0.29: 'I Don't Believe You've Met My Baby' is from the album 'Slide Rule' by Jerry Douglas.
  Fact 0.30: 'In the Palm of Your Hand' was written by Ron Block.
  Fact 0.31: 'In the Palm of Your Hand' runs 3 minutes 24 seconds.
  Fact 0.32: 'In the Palm of Your Hand' is from the album 'I Know Who Holds Tomorrow' by Alison Krauss & The Cox Family.
  Fact 0.33: 'When You Say Nothing at All' was written by Paul Overstreet and Don Schlitz.
  Fact 0.34: 'When You Say Nothing at All' runs 4 minutes 21 seconds.
  Fact 0.35: 'When You Say Nothing at All' is performed by Alison Krauss & Union Station on the album 'Keith Whitley: A Tribute Album'.

Retrieved items:
<item 1>
Now That I've Found You: A Collection
Brent Truitt : Mandolin Stuart Duncan : Fiddle , Harmonica , Cymbals Adam Steffey : Mandolin & Mandola Pete Wasner : Piano Chart performance ( edit ) Album (
edit ) Chart ( 1995 ) Peak position Canadian RPM Country Albums U.S. Billboard 200 13 U.S. Billboard Top Country Albums References ( edit ) Jump up ^ Allmusic
review Jump up ^ Robert Christgau review Jump up ^ Rolling Stone review ^ Jump up to : Alison Krauss Artist Chart Position at Billboard Now That I 've Found You
for Rounder Records . Retrieved August 28 , 2010 . Alison Krauss bio for CMT.com . Retrieved June 7 , 2006 . Now That I 've Found You on Amazon . Retrieved June
7 , 2006 . Preceded by The Woman in Me by Shania Twain RPM Country Albums number - one album October 30 - November 5 ,
</item 1>

<item 2>
Now That I've Found You: A Collection
Now That I 've Found You : a Collection - wikipedia Now That I 've Found You : a Collection Jump to : navigation , search Now That I 've Found You : A
Collection Compilation album by Alison Krauss Released February 7 , 1995 Genre Bluegrass / Country Length 42 : 31 Label Rounder Producer Alison Krauss & Union
Station Alison Krauss chronology I Know Who Holds Tomorrow ( 1994 ) I Know Who Holds Tomorrow 1994 Now That I 've Found You : A Collection ( 1995 ) So Long So
Wrong ( 1997 ) So Long So Wrong 1997 Singles from Now That I 've Found You : A Collection `` Baby , Now That I 've Found You '' Released : 1995 `` Oh , Atlanta
'' Released : 1995 Professional ratings Review scores Source Rating Allmusic Robert Christgau A − Rolling Stone Now That I 've Found You : A Collection is an
album by Alison Krauss , released February 7 , 1995 . It is a retrospective of the early part of Krauss ' recording career .
</item 2>

<item 3>
Now That I've Found You: A Collection
Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say
Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' (
Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 (
From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by
Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by
The Cox Family ) `` I Will '' ( John Lennon ,
</item 3>

<item 4>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Now That I've Found You: A Collection
Close and Personal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Now_That_I%27ve_Found_You:_A_Collection&oldid=808197945 '' Categories : 1995
compilation albums Alison Krauss & Union Station albums Rounder Records compilation albums Hidden categories : Articles with hAudio microformats Music infoboxes
with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 1 November 2017 , at 13 : 34 . About Wikipedia
</item 5>

<item 6>
Now That I've Found You: A Collection
1995 Succeeded by Starting Over by Reba McEntire ( hide ) Alison Krauss & Union Station Alison Krauss Barry Bales Ron Block Jerry Douglas Dan Tyminski John
Pennell Jeff White Brent Truitt Mike Harman Adam Steffey Tim Stafford Studio albums Different Strokes Too Late to Cry Two Highways I 've Got That Old Feeling
Every Time You Say Goodbye I Know Who Holds Tomorrow So Long So Wrong Forget About It New Favorite Lonely Runs Both Ways Paper Airplane Windy City Compilation
albums Now That I 've Found You : A Collection Live A Hundred Miles or More : A Collection Essential Alison Krauss With Robert Plant Raising Sand Singles ``
When You Say Nothing at All '' `` Baby Now That I 've Found You '' `` Baby Mine '' `` The Lucky One '' `` Missing You '' `` Please Read the Letter '' `` Rich
Woman '' Featured singles `` Teach Your Children '' `` Somewhere in the Vicinity of the Heart '' `` It 's Not Over '' `` Buy Me a Rose '' `` Whiskey Lullaby ''
Related articles Discography Awards and nominations Alison Brown Robert Plant Viktor Krauss Up !
</item 6>

<item 7>
Now That I've Found You: A Collection
It is a retrospective of the early part of Krauss ' recording career . It includes songs that appeared on her solo albums , albums by Alison Krauss & Union
Station , and some that appeared on an album by Alison Krauss & the Cox Family . It also contains some new material . The album features the hit country single
`` When You Say Nothing at All '' , sold over two million copies earning double platinum RIAA certification . The title single `` Baby , Now That I 've Found
You '' won the Grammy Award for Best Female Country Vocal Performance and `` When You Say Nothing at All '' was named Single of the Year by the Country Music
Association . Contents ( hide ) 1 Track listing 2 Personnel 3 Chart performance 3.1 Album 4 References Track listing ( edit ) `` Baby , Now That I 've Found You
'' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh ,
</item 7>

<item 8>
Now That I've Found You: A Collection
Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the
Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison
Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album ''
performed by Alison Krauss & Union Station ) Personnel ( edit ) Alison Krauss : Vocal , Fiddle , Viola Ron Block : Vocal , Acoustic & Electric Guitars , Banjo
Russ Barenberg , Mike Marshall , Scott Nygaard , Tim Stafford : Guitars Jerry Douglas : Guitars , Dobro Sam Bush , Brent Truitt : Mandolin Stuart Duncan :
Fiddle , Harmonica ,
</item 8>

<item 9>
For What It's Worth
I Hope You Find It '' `` Take It Like a Man '' `` I Walk Alone '' Guest singles `` A Love Like Yours ( Do n't Come Knocking Everyday ) '' `` Dead Ringer for
Love '' `` I Got You Babe '' `` It Ai n't Necessarily So '' `` Love Can Build a Bridge '' `` Più che puoi '' `` Bewitched , Bothered and Bewildered '' Book
Portal Haley Reinhart Discography Videography Songs Studio albums Listen Up ! ( 2012 ) Better ( 2016 ) What 's That Sound ? ( 2017 ) EPs American Idol Season 10
Highlights : Haley Reinhart ( 2011 ) Songs `` Baby , It 's Cold Outside '' `` Free '' `` Ca n't Help Falling in Love '' `` Better '' Concert tours American
Idols LIVE !
</item 9>

<item 10>
Graph query row:
work: Now That I've Found You: A Collection
release_date: February 7, 1995
work_kind: album
relationship: RELATED_TO
related_entity: Rounder
entity_types: ['__KGBuilder__', '__Entity__', 'Organization']
related_date: None
related_kind: Label
related_title: None
role: None
context: None
relationship_detail: None
relation_detail: None
fact: None
result: None
outcome: None
</item 10>
=== END ITEM id=16b04d8afaf6b7e8 ===

=== ITEM id=d2b28e4f92697fb5 ===
Gold sentences and facts:
Sentence 0: Shortwave radio is radio transmission using shortwave radio frequencies . There is no official definition of the band , but the range always includes all of the high frequency band ( HF ) , and generally extends from 1.7 -- 30 MHz ( 176.3 -- 10.0 m ) ; from the high end of the medium frequency band ( MF ) just above the mediumwave AM broadcast band , to the end of the HF band .
  Fact 0.0: Shortwave radio is radio transmission using shortwave radio frequencies.
  Fact 0.1: There is no official definition of the shortwave band.
  Fact 0.2: The shortwave band range always includes the high frequency band (HF).
  Fact 0.3: The shortwave band generally extends from 1.7 to 30 MHz (176.3 to 10.0 m).
  Fact 0.4: The shortwave band extends from the high end of the medium frequency band (MF), just above the mediumwave AM broadcast band.
  Fact 0.5: The shortwave band extends to the end of the HF band.
Sentence 1: Very little infrastructure is required for long - distance two - way communications using shortwave radio . All one needs is a pair of transceivers , each with an antenna , and a source of energy ( such as a battery , a portable generator , or the electrical grid ) . This makes shortwave radio one of the most robust means of communications , which can be disrupted only by interference or bad ionospheric conditions . Modern digital transmission modes such as MFSK and Olivia are even more robust , allowing successful reception of signals well below the noise floor of a conventional receiver .
  Fact 1.0: Very little infrastructure is required for long-distance two-way communication using shortwave radio.
  Fact 1.1: All one needs for shortwave radio communication is a pair of transceivers, each with an antenna.
  Fact 1.2: All one needs for shortwave radio communication is a source of energy, such as a battery, a portable generator, or the electrical grid.
  Fact 1.3: Shortwave radio is one of the most robust means of communications.
  Fact 1.4: Shortwave radio communications can be disrupted only by interference or bad ionospheric conditions.
  Fact 1.5: Modern digital transmission modes such as MFSK and Olivia are even more robust.
  Fact 1.6: Modern digital transmission modes such as MFSK and Olivia allow successful reception of signals well below the noise floor of a conventional receiver.

Retrieved items:
<item 1>
Shortwave radio
See also ( edit ) ALLISS -- a very large rotatable antenna system used in international broadcasting Amateur radio -- also uses the shortwave bands , but with
power levels under 2 kW International broadcasting List of American shortwave broadcasters List of shortwave radio broadcasters Long wave Medium wave Shortwave
bands -- shortwave spectrum allocation shortwave relay station -- the fundamental way in which programmes are broadcast on shortwave SSB -- a method of radio
signal modulation References ( edit ) Jump up ^ Grundig Satellit 400 international / professional ShortWaveRadio.ch Jump up ^ Inconsistent article :
itwissen.info : KW ( Kurzwelle ) SW ( short wave ) Quote : `` ... Der KW - Frequenzbereich ( SW ) liegt zwischen 3 MHz und 30 MHz ( < - 1 . definition ) ...
Kurzwelle ist auch eine sendetechnische Bezeichnung für Rundfunk im Frequenzbereich zwischen 2 , 3 MHz und 26 , 1 MHz ( < - 2 .
</item 1>

<item 2>
Shortwave radio
The drawback is the receiver is more complicated , since it must re-create the carrier to recover the signal . Small errors in the detection process greatly
affect the pitch of the received signal . As a result , single sideband is not used for music or general broadcast . Single sideband is used for long - range
voice communications by ships and aircraft , Citizen 's Band , and amateur radio operators . Lower sideband ( LSB ) is generally used below 9 MHz and USB (
upper sideband ) above 9 MHz . Vestigal sideband transmits the carrier and one complete sideband , but filters out the other sideband . It is a compromise
between AM and SSB , enabing simple receivers to be used , but requires almost as much transmitter power as AM . One advantage is only half the bandwidth of an
AM signal is used . It can be heard in the transmission of certain radio time signal stations . Vestigial sideband is used for over the air Television
Broadcasts both analog and digital .
</item 2>

<item 3>
Shortwave radio
The last WRC took place in 2007 . At WRC - 97 in 1997 , the following bands were allocated for international broadcasting . AM shortwave broadcasting channels
are allocated with a 5 kHz separation for traditional analog audio broadcasting . Metre Band Frequency Range Remarks 120 m 2.3 -- 2.495 MHz tropical band 90 m
3.2 -- 3.4 MHz tropical band 75 m 3.9 -- 4 MHz shared with the North American amateur radio 80m band 60 m 4.75 -- 5.06 MHz tropical band 49 m 5.9 -- 6.2 MHz 41
m 7.2 -- 7.6 MHz shared with the amateur radio 40m band 31 m 9.4 -- 9.9 MHz currently the most heavily used band 25 m 11.6 -- 12.2 MHz 22 m 13.57 -- 13.87 MHz
19 m 15.1 -- 15.8 MHz 16 m 17.48 -- 17.9 MHz 15 m 18.9 -- 19.02 MHz almost unused ,
</item 3>

<item 4>
Shortwave radio
Continuous wave ( CW ) is on - and - off keying of a carrier , used for Morse code communications and Hellschreiber facsimile - based teleprinter transmissions
. Narrow - band frequency modulation ( NBFM ) is used typically above 20 MHz . Because of the larger bandwidth required , NBFM is commonly used for VHF
communication . Regulations limit the bandwidth of a signal transmitted in the HF bands , and the advantages of frequency modulation are greatest if the FM
signal has a wide bandwidth . NBFM is limited to short - range transmissions due to the multiphasic distortions created by the ionosphere . Digital Radio
Mondiale ( DRM ) is a digital modulation for use on bands below 30 MHz . Radioteletype , fax , digital , slow - scan television and other systems use forms of
frequency - shift keying or audio subcarriers on a shortwave carrier . These generally require special equipment to decode , such as software on a computer
equipped with a sound card .
</item 4>

<item 5>
Shortwave radio
Shortwave radio received its name because the wavelengths in this band are relatively shorter than 200 m ( 1,500 kHz ) which marked the original upper limit of
the medium frequency band first used for radio communications . The broadcast medium wave band now extends above the 200 m / 1,500 kHz limit , and the amateur
radio 1.8 MHz -- 2.0 MHz band ( known as the `` top band '' ) is the lowest - frequency band considered to be ' shortwave ' . Early long distance radio
telegraphy used long waves , below 300 kilohertz ( kHz ) . The drawbacks to this system included a very limited spectrum available for long distance
communication , and the very expensive transmitters , receivers and gigantic antennas that were required . It was also difficult to beam the radio wave
directionally with long wave , resulting in a major loss of power over long distances . Prior to the 1920s , the shortwave frequencies above 1.5 MHz were
regarded as useless for long distance communication and were designated in many countries for amateur use .
</item 5>

<item 6>
Shortwave radio
L band S band Digital systems ADR DAB - S DVB - SH S - DMB SDR Commercial radio providers 1worldspace Sirius XM Holdings SiriusXM Canada Codecs AAC AMR - WB+
HDC HE - AAC MPEG - 1 Audio Layer II Subcarrier signals AMSS DirectBand PAD RDS / RBDS SCA / SCMO DARC Related topics Technical ( audio ) Audio data compression
Audio signal processing Technical ( AM stereo formats ) Belar C - QUAM Harris Kahn - Hazeltine Magnavox Technical ( emission ) AM broadcasting AM expanded band
Cable radio Digital radio Error detection and correction FM broadcast band FM broadcasting Multipath propagation Shortwave relay station Cultural History of
radio International broadcasting Comparison of radio systems Radio spectrum ( ITU ) ELF 3 Hz / 100 Mm 30 Hz / 10 Mm SLF 30 Hz / 10 Mm 300 Hz / 1 Mm ULF 300 Hz /
1 Mm 3 kHz / 100 km VLF 3 kHz / 100 km 30 kHz / 10 km LF 30 kHz / 10 km 300 kHz / 1 km MF 300 kHz / 1 km 3 MHz / 100 m HF
</item 6>

<item 7>
Shortwave radio
`` Short - Wave Radio Telephone is Success in Tests '' Popular Mechanics , July 1931 , mid page experiments carried out for the French and British governments
Que Escuchar en la Onda Corta en Español website Time signal stations Longwave BPC BPL BSF DCF77 JJY RBU RJH66 ( Beta ) RTZ MSF TDF WWVB Shortwave BPM BSF CHU
HD2IOA HLA JN53DV RWM ROA WWV WWVH ( Hawaii , USA ) YVTO VHF / FM / UHF RDS CT Satellite GPS BeiDou Galileo GLONASS IRNSS QZSS Defunct HBG NAA WWVL OMA OLB5 VNG
Y3S Analog and digital audio broadcasting Terrestrial Radio modulation AM FM COFDM Frequency allocations LW ( LF ) MW ( MF ) SW ( HF ) VHF ( low / mid / high )
L band ( UHF ) Digital systems CAM - D DAB / DAB+ DRM / DRM+ FMeXtra HD Radio CDR DVB - T2 Lite Satellite Frequency allocations C band K band L band S band
Digital systems ADR DAB - S DVB - SH S - DMB
</item 7>

<item 8>
Shortwave radio
Users ( edit ) Portable shortwave receiver 's digital display tuned to the 75 meter band Some established users of the shortwave radio bands may include :
International broadcasting primarily by government - sponsored propaganda , international news ( for example , the BBC World Service ) or cultural stations to
foreign audiences : the most common use of all . Domestic broadcasting : to widely dispersed populations with few longwave , mediumwave and FM stations serving
them ; or for specialty political , religious and alternative media networks ; or of individual commercial and non-commercial paid broadcasts . Oceanic air
traffic control uses the HF / shortwave band for long distance communication to aircraft over the oceans and poles , which are far beyond the range of
traditional VHF frequencies . Modern systems also include satellite communications , such as ADS - C / CPDLC `` Utility '' stations transmitting messages not
intended for the general public , such as merchant shipping , marine weather , and ship - to - shore stations ; for aviation weather and air - to - ground
communications ; for military communications ; for long - distance governmental purposes , and for other non-broadcast communications .
</item 8>

<item 9>
Shortwave radio
Contents ( hide ) 1 Frequency classifications 2 History 2.1 Development 2.2 Amateur use of shortwave propagation 3 Propagation characteristics 4 Types of
modulation 5 Users 6 Shortwave broadcasting 6.1 Frequency allocations 6.2 Advantages 6.3 Disadvantages 7 Shortwave listening 8 Shortwave broadcasts and music 9
Shortwave 's future 10 See also 11 References 12 External links Frequency classifications ( edit ) The widest popular definition of the shortwave frequency
interval is the ITU Region 1 ( EU + Africa + Russia ... ) definition , and is the span 1.6 -- 30 MHz , just above the medium wave band , which ends
approximately at 1.6 MHz .
</item 9>

<item 10>
Shortwave radio
Solar flares produce a large increase in D region ionization so high , sometimes for periods of several minutes , all skywave propagation is nonexistent . Types
of modulation ( edit ) Further information : Modulation National Panasonic R3000 analog shortwave receiver , circa 1965 Several different types of modulation
are used to impress information on a short - wave transmission . Amplitude modulation is the simplest type and the most commonly used for shortwave broadcasting
. The instantaneous amplitude of the carrier is controlled by the amplitude of the signal ( speech , or music , for example ) . At the receiver , a simple
detector recovers the desired modulation signal from the carrier . Single sideband transmission is a form of amplitude modulation but in effect filters the
result of modulation . An amplitude - modulated signal has frequency components both above and below the carrier frequency . If one set of these components is
eliminated as well as the residual carrier , only the remaining set is transmitted . This reduces power in the transmission , as roughly 2 / 3 of the energy
sent by an AM signal is unnecessary to recover the information contained on it . It also reduces `` bandwidth '' , enabling about one - half the carrier
frequency spectrum width to be used .
</item 10>
=== END ITEM id=d2b28e4f92697fb5 ===

=== ITEM id=475673b95efadedc ===
Gold sentences and facts:
Sentence 0: In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers . When Maximoff throws the explosion into the sky with telekinesis , it damages a nearby building , killing several Wakandan humanitarian workers .
  Fact 0.0: In 1991, the brainwashed super-soldier James 'Bucky' Barnes was dispatched from a Hydra base in Siberia.
  Fact 0.1: In 1991, James 'Bucky' Barnes was dispatched to intercept an automobile carrying a case of super-soldier serum.
  Fact 0.2: In the present day, Ultron's defeat in the nation of Sokovia at the hands of the Avengers happened approximately one year earlier.
  Fact 0.3: Ultron was defeated in the nation of Sokovia by the Avengers.
  Fact 0.4: In the present day, Steve Rogers, Natasha Romanoff, Sam Wilson, and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos.
  Fact 0.5: Brock Rumlow blows himself up.
  Fact 0.6: Brock Rumlow hoped to kill Steve Rogers by blowing himself up.
  Fact 0.7: Wanda Maximoff throws the explosion into the sky with telekinesis.
  Fact 0.8: The explosion damages a nearby building.
  Fact 0.9: The explosion killed several Wakandan humanitarian workers.

Retrieved items:
<item 1>
Captain America: Civil War
In one program , Everhart discusses events leading to Captain America : Civil War . The post-credits scene for Ant - Man featured footage shot by the Russo
brothers for Captain America : Civil War , showing Wilson and Rogers with Barnes in their custody , and unable to contact Stark because of `` the accords '' ;
Wilson mentions that he `` know ( s ) a guy '' , implying Lang . Feige explained that when the post-credit sequence appears in Civil War it may appear with ``
different takes ... different angles . '' Stan said the scene was shot in early May 2015 , and would appear in the middle of Civil War . The first footage of
the film debuted in August 2015 at the D23 Expo . Additional footage was shown in September 2015 at the Asia Pop Comic Convention . The footage shown at the D23
Expo and 2015 Asia Pop Comic Convention received an overwhelmingly positive response from the audiences . The first trailer for the film debuted on Jimmy Kimmel
Live ! on November 25 , 2015 , and within hours became the number one trending topic on Twitter .
</item 1>

<item 2>
Captain America: Civil War
We wanted to figure out very personal reasons why everyone 's relationship to this idea of registration is going to become complicated . That 's what the
relationship between Steve and Bucky allowed us to do , to get very personal in terms of why people would lean one way or the other . '' Executive producer Nate
Moore added that `` it felt like it was kind of the right time '' to adapt `` Civil War '' given The Avengers , plus many of the Phase Two films ( Thor : The
Dark World , The Winter Soldier , and Avengers : Age of Ultron ) , all dealt with `` world - ending experiences . We felt like we had to tell the next step in
that story , which is ... what is the world 's reaction ? '' In November 2014 , Daniel Brühl joined the cast in an unspecified role , while Anthony Mackie and
Frank Grillo were confirmed to return as Sam Wilson / Falcon and Brock Rumlow / Crossbones , respectively .
</item 2>

<item 3>
Captain America: Civil War
In August 2014 , the Russos stated that the film would be set `` a couple years '' after The Winter Soldier , and would continue to focus on Steve Rogers '
relationship with Bucky Barnes as well as the political themes related to Captain America . Anthony stated , `` The character was invented for an explicitly
political purpose . So it 's hard to get away from that nature . '' The Russos also said that they would be `` bringing some new elements to the table that will
give us a twist on Winter Soldier '' , and indicated that filming was scheduled to begin in Atlanta . They described themselves as `` ecstatic '' with a first
draft of the screenplay submitted by Markus and McFeely , and also stated that the film 's title would be announced `` in a month or so at most '' , and that
the concept and title for the film came from Feige , who had it `` for a while '' . In September , Joe expanded by saying the film would have another `` big
idea that alters the universe as a whole in some way '' similar to S.H.I.E.L.D. falling in The Winter Soldier .
</item 3>

<item 4>
Captain America: Civil War
It is the sequel to 2011 's Captain America : The First Avenger and 2014 's Captain America : The Winter Soldier , and the thirteenth film in the Marvel
Cinematic Universe ( MCU ) . The film is directed by Anthony and Joe Russo , with a screenplay by Christopher Markus & Stephen McFeely , and features an
ensemble cast , including Chris Evans , Robert Downey Jr. , Scarlett Johansson , Sebastian Stan , Anthony Mackie , Don Cheadle , Jeremy Renner , Chadwick
Boseman , Paul Bettany , Elizabeth Olsen , Paul Rudd , Emily VanCamp , Tom Holland , Frank Grillo , William Hurt , and Daniel Brühl . In Captain America : Civil
War , disagreement over international oversight of the Avengers fractures them into opposing factions -- one led by Steve Rogers and the other by Tony Stark .
Development of Civil War began in late 2013 when Markus and McFeely began writing the screenplay , which borrows concepts from the 2006 comic book storyline ``
Civil War '' , while also focusing on story and character elements from the previous Captain America films to conclude the trilogy .
</item 4>

<item 5>
Captain America: Civil War
The next month , it was revealed that the film would be converted to 3D in post-production , and that Brühl would be playing Helmut Zemo . Additionally ,
Elizabeth Olsen revealed she would reprise her role in the film as Wanda Maximoff / Scarlet Witch . Filming ( edit ) Film set for Captain America : Civil War in
Downtown Atlanta Principal photography began on April 27 , 2015 , in Fayette County , Georgia at Pinewood Atlanta Studios , under the working title Sputnik .
Other filming locations in the Atlanta metropolitan area included the Buckhead district of Atlanta ; the Peachtree Christian Church in Midtown Atlanta ;
Downtown Atlanta , including an area near Philips Arena known as The Gulch , which served as the market in Lagos ; Norcross , Georgia ; Porsche 's headquarters
in Atlanta at Aerotropolis Atlanta , which served as the Avengers ' headquarters in the film ; and the Atlanta Civic Center , which served as the IFID (
Institute for Infectious Diseases ) laboratory in Lagos ; the latter 's interiors doubled as MIT , and a third part was used for a Berlin location .
</item 5>

<item 6>
Captain America: Civil War
Contents ( hide ) 1 Plot 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 3.4. 1 Visual effects 4 Music 5 Release 5.1
Marketing 5.1. 1 Merchandise 5.2 Home media 6 Reception 6.1 Box office 6.1. 1 United States and Canada 6.1. 2 Other territories 6.2 Critical response 6.3
Accolades 7 Future 8 Notes 9 References 10 External links Plot ( edit ) In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a
Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in
the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a
biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers .
</item 6>

<item 7>
Captain America: Civil War
Following the November 2014 hacking of Sony Pictures ' computers , emails between Sony Pictures Entertainment co-chairman Amy Pascal and president Doug Belgrad
were released stating that Marvel wanted to include Spider - Man ( whose film rights are licensed to Sony ) in the film , but talks between the studios
concerning this were believed to have broken down . However , in February 2015 , Sony Pictures and Marvel Studios reached a licensing deal for the use of Spider
- Man in an MCU film , and reports indicated that the character would indeed appear in Civil War . The Russos stated they were lobbying for months to include
the character in the film . In January 2015 , Mackie revealed that , in addition to Atlanta , filming locations would include Puerto Rico and Berlin , while the
Russo brothers confirmed that Scarlett Johansson would return in the film as Natasha Romanoff / Black Widow . Editor Jeffrey Ford , who worked on The Winter
Soldier , also signed on for Civil War . In March 2015 , Jeremy Renner was revealed to be reprising his role as Clint Barton / Hawkeye .
</item 7>

<item 8>
Captain America: Civil War
Captain America : Civil War - Wikipedia Captain America : Civil War Jump to : navigation , search Captain America : Civil War Theatrical release poster Directed
by Anthony Russo Joe Russo Produced by Kevin Feige Screenplay by Christopher Markus Stephen McFeely Based on Captain America by Joe Simon Jack Kirby Starring
Chris Evans Robert Downey Jr . Scarlett Johansson Sebastian Stan Anthony Mackie Don Cheadle Jeremy Renner Chadwick Boseman Paul Bettany Elizabeth Olsen Paul
Rudd Emily VanCamp Tom Holland Frank Grillo William Hurt Daniel Brühl Music by Henry Jackman Cinematography Trent Opaloch Edited by Jeffrey Ford Matthew Schmidt
Production company Marvel Studios Distributed by Walt Disney Studios Motion Pictures Release date April 12 , 2016 ( 2016 - 04 - 12 ) ( Dolby Theatre ) May 6 ,
2016 ( 2016 - 05 - 06 ) ( United States ) Running time 147 minutes Country United States Language English Budget $250 million Box office $1.153 billion Captain
America : Civil War is a 2016 American superhero film based on the Marvel Comics character Captain America , produced by Marvel Studios and distributed by Walt
Disney Studios Motion Pictures .
</item 8>

<item 9>
Captain America: Civil War
Following positive reactions to test screenings of The Winter Soldier , the Russo brothers were brought back to direct in early 2014 . The film 's title and
premise were revealed in October 2014 , along with Downey 's casting ; additional cast members joined in the following months . Principal photography began in
April 2015 at Pinewood Atlanta Studios in Fayette County , Georgia , and continued in the Metro Atlanta area before concluding in Germany in August 2015 , with
the film being the first to use IMAX 's digital 2D cameras ( for the film 's central airport fight sequence ) . Visual effects were provided by nearly 20
different studios during the post-production process . Captain America : Civil War held its world premiere in Los Angeles on April 12 , 2016 , and was released
in the United States on May 6 , 2016 , in 3D and IMAX 3D . The film became a critical and commercial success , grossing over $1.1 billion worldwide , making it
the highest - grossing film of 2016 and , at the time , the 12th - highest - grossing film ever .
</item 9>

<item 10>
Captain America: Civil War
wikipedia.org/w/index.php?title=Captain_America:_Civil_War&oldid=819487324 '' Categories : 2016 films English - language films 2010s action thriller films 2010s
adventure films 2010s psychological thriller films 2010s science fiction films 2010s superhero films 2016 3D films American 3D films American action thriller
films American films American political thriller films American psychological thriller films American science fiction action films American science fiction
thriller films American sequel films Babelsberg Studio films Captain America films Crossover films Drone films Films scored by Henry Jackman Films about revenge
Films about terrorism Films based on works by Mark Millar Films directed by Anthony and Joe Russo Films set in 1991 Films set in Berlin Films set in Bucharest
Films set in Cleveland Films set in Germany Films set in Lagos Films set in London Films set in Massachusetts Films set in New York ( state ) Films set in
Queens , New York Films set in Siberia Films set in Vienna Films set in prison Films set in a fictional African country Films shot at Pinewood Atlanta Studios
Films shot in Atlanta Films shot in Berlin Films shot in Brazil Films shot in Germany Films shot in Norway Films shot
</item 10>
=== END ITEM id=475673b95efadedc ===

=== ITEM id=0d444366ea75dd17 ===
Gold sentences and facts:
Sentence 0: Symphony No. 40 in G minor , KV. 550 was written by Wolfgang Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 . The two are the only extant minor key symphonies Mozart wrote .
  Fact 0.0: Symphony No. 40 in G minor, KV. 550 was written by Wolfgang Amadeus Mozart.
  Fact 0.1: Symphony No. 40 in G minor, KV. 550 was written in 1788.
  Fact 0.2: Symphony No. 40 in G minor, KV. 550 is sometimes referred to as the 'Great G minor symphony'.
  Fact 0.3: The Great G minor symphony is distinguished from the 'Little G minor symphony', No. 25.
  Fact 0.4: Symphony No. 40 and the Little G minor symphony, No. 25, are the only extant minor key symphonies Mozart wrote.

Retrieved items:
<item 1>
Symphony No. 40 (Mozart)
Reception ( edit ) This work has elicited varying interpretations from critics . Robert Schumann regarded it as possessing `` Grecian lightness and grace '' .
Donald Francis Tovey saw in it the character of opera buffa . Almost certainly , however , the most common perception today is that the symphony is tragic in
tone and intensely emotional ; for example , Charles Rosen ( in The Classical Style ) has called the symphony `` a work of passion , violence , and grief '' .
Although interpretations differ , the symphony is unquestionably one of Mozart 's most greatly admired works , and it is frequently performed and recorded .
Influence ( edit ) Ludwig van Beethoven knew the symphony well , copying out 29 bars from the score in one of his sketchbooks . As Gustav Nottebohm observed in
1887 , the copied bars appear amid the sketches for Beethoven 's Fifth Symphony , whose third movement begins with a pitch sequence similar to that of Mozart 's
finale ( see example above ) .
</item 1>

<item 2>
Symphony No. 40 (Mozart)
Nikolaus Harnoncourt conjectured that Mozart composed the three symphonies as a unified work , pointing , among other things , to the fact that the Symphony No.
40 , as the middle work , has no introduction ( unlike No. 39 ) and does not have a finale of the scale of No. 41 's . The 40th symphony exists in two versions
, differing primarily in that one includes parts for a pair of clarinets ( with suitable adjustments made in the other wind parts ) . Most likely , the clarinet
parts were added in a revised version . The autograph scores of both versions were acquired in the 1860s by Johannes Brahms , who later donated the manuscripts
to the Gesellschaft der Musikfreunde in Vienna , where they reside today . Premiere ( edit ) As Neal Zaslaw has pointed out , writers on Mozart have often
suggested -- or even asserted -- that Mozart never heard his 40th Symphony performed . Some commentators go further , suggesting that Mozart wrote the symphony
( and its companions , Nos.
</item 2>

<item 3>
Symphony No. 40 (Mozart)
Zaslaw , Neal , and William Cowdery ( 1990 ) . The Compleat Mozart : A Guide to the Musical Works of Wolfgang Amadeus Mozart . New York : W.W. Norton & Company
. Extracts cited are viewable online at Google Books : ( 6 ) . External links ( edit ) Wikimedia Commons has media related to Symphony No. 40 ( Mozart ) .
Versions of the score : Sinfonie in g minor K. 550 ( 1st version ) : Score and critical report ( in German ) in the Neue Mozart - Ausgabe Sinfonie in g minor K.
550 ( 2nd version ) : Score in the Neue Mozart - Ausgabe Symphony No. 40 : Scores at the International Music Score Library Project ( IMSLP ) Kalmus miniature
editions , from the William and Gayle Cook Music Library at the Indiana University Jacobs School of Music A Public Domain version by the Columbia University
Orchestra at TEOC Classical Music Radio : Symphony No.
</item 3>

<item 4>
Symphony No. 40 (Mozart)
Jump up ^ Milada Jonášová : `` Eine Aufführung der g - moll - Sinfonie KV 550 bei Baron van Swieten im Beisein Mozarts '' , in : Mozart Studien 20 , Tutzing
2011 , pp. 253 -- 268 . An abridged English translation was published in the Newsletter of the Mozart Society of America 16.1 ( 2012 ) , currently available on
line at ( 2 ) . The letter reads `` im Wien habe ich selbst es von verstorbenem Mozart gehört , als Er sie bei Baron Wanswiten ( sic ) hat produciren lassen ,
das ( s ) er wärend der production aus dem Zimmer sich hat entfernen müssen , wie man Sie unrichtig aufgeführt hat '' , meaning `` and in Vienna I have heard
myself from the departed Mozart , that when he had it performed in Baron Wanswiten 's rooms , he had to leave the room during the performance because it was
being played so incorrectly '' . Jump up ^ List from Zaslaw 1983 .
</item 4>

<item 5>
Symphony No. 40 (Mozart)
504 , Prague ) No. 39 ( K. 543 ) No. 40 ( K. 550 ) No. 41 ( K. 551 , Jupiter ) Lost K. 19b K. 66c K. 66d K. 66e K. Anh. C 11.07 K. Anh. C 11.08 No. 2 now
attributed to Leopold Mozart . No. 3 now attributed to Carl Friedrich Abel . No. 37 now attributed to Michael Haydn . List of symphonies by Wolfgang Amadeus
Mozart List of spurious / doubtful Mozart symphonies LCCN : n81018421 GND : 300111150 BNF : cb13915276b ( data ) Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Symphony_No._40_(Mozart)&oldid=825476175 '' Categories : Symphonies by Wolfgang Amadeus Mozart 1788 compositions
Compositions in G minor Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked weasel - worded
phrases from
</item 5>

<item 6>
Symphony No. 40 (Mozart)
Franz Schubert likewise copied down the music of Mozart 's minuet , and the minuet of his Fifth Symphony strongly evokes Mozart 's . Zaslaw has suggested that a
passage late in Joseph Haydn 's oratorio The Seasons ( 1801 ) , a meditation on death , quotes the second movement of the 40th Symphony and was included by
Haydn as a memorial to his long - dead friend . First recording ( edit ) The first known recording of the 40th Symphony is by the Victor Recording Company and
issued in 1915 under the title `` Symphony in G Minor '' . The Victor Concert Orchestra performed under the direction of conductor Walter B. Rogers . Media (
edit ) The following files contain a digital recording of a performance of the 40th Symphony by the Fulda Symphonic Orchestra . The performance took place on
March 18 , 2001 in the Orangerie in Fulda , Germany . 1st movement Molto allegro 2nd movement Andante 3rd movement Menuetto , Allegretto - Trio 4th movement
Allegro assai Problems playing these files ? See media help .
</item 6>

<item 7>
Symphony No. 40 (Mozart)
The work is in four movements , in the usual arrangement for a classical - style symphony ( fast movement , slow movement , minuet , fast movement ) : Molto
allegro , Andante , Menuetto . Allegretto -- Trio , Finale . Allegro assai , The first movement begins darkly , not with its first theme but with accompaniment
, played by the lower strings with divided violas . The technique of beginning a work with an accompaniment figure was later used by Mozart in his last piano
concerto ( KV. 595 ) and later became a favourite of the Romantics ( examples include the openings of Mendelssohn 's Violin Concerto and Sergei Rachmaninoff 's
Third Piano Concerto ) . The first theme is as follows . The second movement is a lyrical work in time . It is in the subdominant key of the relative major of G
minor ( B ♭ major ) : E ♭ major .
</item 7>

<item 8>
Symphony No. 40 (Mozart)
The contrapuntal opening bars of this movement appear thus in keyboard reduction : The minuet begins with an angry , cross-accented hemiola rhythm and a pair of
three - bar phrases , as shown in the following piano reduction : Various commentators have asserted that while the music is labelled `` minuet '' , it would
hardly be suitable for dancing . The contrasting gentle trio section , in G major , alternates the playing of the string section with that of the winds . The
fourth movement opens with a series of rapidly ascending notes outlining the tonic triad illustrating what is commonly referred to as the Mannheim rocket . The
movement is written largely in eight - bar phrases , following the general tendency toward rhythmic squareness in the finales of classical - era symphonies . A
remarkable modulating passage in which every tone in the chromatic scale but one is played , strongly destabilising the key , occurs at the beginning of the
development section . The single note left out is in fact a G ( the tonic ) . The symphony typically has a duration of about 25 minutes . Reception ( edit )
This work has elicited varying interpretations from critics .
</item 8>

<item 9>
Symphony No. 40 (Mozart)
334 . Jump up ^ Hopkins 1981 . Jump up ^ Nottebohm 1887 , p. 531 . Jump up ^ Zaslaw and Cowdery 1990 , p. 210 . Jump up ^ See No. 38 , the aria `` Erblicke hier
, bethörter Mensch '' . Source : Zaslaw and Cowdery 1990 , p. 210 ) . Jump up ^ `` Wolfgang Amadeus Mozart - Symphony in G Minor '' . Discography of American
Historical Recordings . References ( edit ) Biba , Otto ( 2009 ) . `` Kommentar '' , Wolfgang Amadeus Mozart . Sinfonie g - moll , KV 550 . Autographe Partitur
. Erste und zweite Fassung . Faksimile - Ausgabe . Gesellschaft der Musikfreunde in Wien , p. 8 . Deutsch , Otto Erich ( 1965 ) . Mozart : A Documentary
Biography . Stanford , CA : Stanford University Press .
</item 9>

<item 10>
Symphony No. 40 (Mozart)
Symphony No. 40 ( Mozart ) - wikipedia Symphony No. 40 ( Mozart ) Jump to : navigation , search Symphony No. 40 in G minor , KV. 550 was written by Wolfgang
Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 .
The two are the only extant minor key symphonies Mozart wrote . Contents ( hide ) 1 Composition 2 Premiere 3 The music 4 Reception 5 Influence 6 First recording
7 Media 8 Notes 9 References 10 External links Composition ( edit ) The date of completion of this symphony is known exactly , since Mozart in his mature years
kept a full catalog of his completed works ; he entered the 40th Symphony into it on 25 July 1788 . Work on the symphony occupied an exceptionally productive
period of just a few weeks during which time he also completed the 39th and 41st symphonies ( 26 June and 10 August , respectively ) .
</item 10>
=== END ITEM id=0d444366ea75dd17 ===

=== ITEM id=5344f31de20a3e29 ===
Gold sentences and facts:
Sentence 0: Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout . He is the only Eagle Scout to have ascended to the U.S. Presidency .
  Fact 0.0: Ford was involved in the Boy Scouts of America.
  Fact 0.1: Ford earned the Boy Scouts of America program's highest rank, Eagle Scout.
  Fact 0.2: Ford is the only Eagle Scout to have ascended to the U.S. Presidency.

Retrieved items:
<item 1>
Gerald Ford
Legacy of Honor : The Values and Influence of America 's Eagle Scouts . New York : St. Martin 's Press . pp. 12 -- 13 and 87 . ISBN 0 - 312 - 36653 - 1 .
Retrieved December 29 , 2006 . Jump up ^ `` Investigatory Records on Gerald Ford , Applicant for a Commission '' ( PDF ) . Gerald R. Ford Presidential Library .
December 30 , 1941 . Retrieved November 18 , 2010 . Jump up ^ Wertheimer , Linda ( December 27 , 2006 ) . `` Special Report : Former President Gerald Ford Dies
; Sought to Heal Nation Disillusioned by Watergate Scandal '' . National Public Radio . Retrieved April 26 , 2009 . Jump up ^ Perry , Will ( 1974 ) . `` No
Cheers From the Alumni '' . The Wolverines : A Story of Michigan Football ( PDF ) . Huntsville , Alabama : The Strode Publishers . pp. 150 -- 152 . ISBN 0 -
87397 - 055 - 1 .
</item 1>

<item 2>
Gerald Ford
Ford was made a 33 ° Scottish Rite Mason on September 26 , 1962 . In April 1975 , Ford was elected by a unanimous vote Honorary Grand Master of the
International Supreme Council , Order of DeMolay , a position in which he served until January 1977 . Ford received the degrees of York Rite Masonry ( Chapter
and Council degrees ) in a special ceremony in the Oval Office on January 11 , 1977 , during his term as President of the United States . Honors ( edit ) Ford
honored on U.S Postage , issue of 2007 Ford received the Distinguished Eagle Scout Award in May 1970 , as well as the Silver Buffalo Award , from the Boy Scouts
of America . In 1974 , he also received the highest distinction of the Scout Association of Japan , the Golden Pheasant Award . In 1985 , he received the 1985
Old Tom Morris Award from the Golf Course Superintendents Association of America , GCSAA 's highest honor . In 1992 , the U.S. Navy Memorial Foundation awarded
Ford its Lone Sailor Award for his naval service and his subsequent government service .
</item 2>

<item 3>
Gerald Ford
That year his biological father , whom Ford described as a `` carefree , well - to - do man who did n't really give a damn about the hopes and dreams of his
firstborn son '' , approached Ford while he was waiting tables in a Grand Rapids restaurant . The two `` maintained a sporadic contact '' until Leslie King ,
Sr. 's death in 1941 . Ford said , `` My stepfather was a magnificent person and my mother equally wonderful . So I could n't have written a better prescription
for a superb family upbringing . '' Eagle Scout Gerald Ford ( circled in red ) in 1929 ; Michigan Governor Fred W. Green at far left , holding hat Ford as a
center on the University of Michigan football team , 1933 Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout
. He is the only Eagle Scout to have ascended to the U.S. Presidency . Ford attended Grand Rapids South High School , where he was a star athlete and captain of
the football team . In 1930 , he was selected to the All - City team of the Grand Rapids City League .
</item 3>

<item 4>
President of the United States
Images of documents written by U.S. presidents Presidents of the United States ( list ) George Washington ( 1789 -- 1797 ) John Adams ( 1797 -- 1801 ) Thomas
Jefferson ( 1801 -- 1809 ) James Madison ( 1809 -- 1817 ) James Monroe ( 1817 -- 1825 ) John Quincy Adams ( 1825 -- 1829 ) Andrew Jackson ( 1829 -- 1837 )
Martin Van Buren ( 1837 -- 1841 ) William Henry Harrison ( 1841 ) John Tyler ( 1841 -- 1845 ) James K. Polk ( 1845 -- 1849 ) Zachary Taylor ( 1849 -- 1850 )
Millard Fillmore ( 1850 -- 1853 ) Franklin Pierce ( 1853 -- 1857 ) James Buchanan ( 1857 -- 1861 ) Abraham Lincoln ( 1861 -- 1865 ) Andrew Johnson ( 1865 --
1869 ) Ulysses S. Grant ( 1869 -- 1877 ) Rutherford B. Hayes ( 1877 -- 1881 ) James A.
</item 4>

<item 5>
President of the United States
Boy Scouts of America . Retrieved November 9 , 2017 . Jump up ^ Grier , Peter ( April 25 , 2011 ) . `` The ( not so ) secret history of the White House Easter
Egg Roll '' . The Christian Science Monitor . Archived from the original on July 30 , 2012 . Retrieved July 30 , 2012 . Jump up ^ Hesse , Monica ( November 21 ,
2007 ) . `` Turkey Pardons , The Stuffing of Historic Legend '' . The Washington Post . Retrieved May 14 , 2011 . Jump up ^ Gibbs , Nancy ( November 13 , 2008 )
. `` How Presidents Pass The Torch '' . Time . Retrieved May 6 , 2011 . Jump up ^ Dorning , Mike ( January 22 , 2009 ) . `` A note from Bush starts morning in
the Oval Office '' . Chicago Tribune . Archived from the original on December 28 , 2011 . Retrieved May 6 , 2011 .
</item 5>

<item 6>
President of the United States
Educational site on the American presidency `` Presidents ' Occupations '' . Retrieved August 20 , 2007 . Listing of every president 's occupations before and
after becoming the commander in chief `` The Masonic Presidents Tour '' . The Masonic Library and Museum of Pennsylvania . Archived from the original on October
12 , 2014 . Retrieved October 7 , 2005 . Brief histories of the Masonic careers of presidents who were members of the Freemasons `` The Presidents '' . American
Experience . Retrieved March 4 , 2007 . PBS site on the American presidency Presidents of the United States : Resource Guides from the Library of Congress
Shapell Manuscript Foundation . Images of documents written by U.S.
</item 6>

<item 7>
Gerald Ford
Bush , Peter Dominick , Robert Griffin , Thomas Kuchel , Mel Laird , Bob Mathias , George Murphy , Dick Poff , Chuck Percy , Al Quie , Charlotte Reid , Hugh
Scott , Bill Steiger , John Tower ( 1968 ) Vacant Title next held by Donald Fraser , Scoop Jackson , Mike Mansfield , John McCormack , Patsy Mink , Ed Muskie ,
Bill Proxmire Preceded by Richard Nixon Republican nominee for President of the United States 1976 Succeeded by Ronald Reagan Political offices Preceded by
Spiro Agnew Vice President of the United States 1973 -- 1974 Succeeded by Nelson Rockefeller Preceded by Richard Nixon President of the United States 1974 --
1977 Succeeded by Jimmy Carter Diplomatic posts Preceded by Valéry Giscard d'Estaing Chair of the Group of Seven 1976 Succeeded by James Callaghan Honorary
titles Preceded by Ronald Reagan Oldest living President of the United States 2004 -- 2006 Succeeded by George H.W.
</item 7>

<item 8>
Gerald Ford
Ford was the only person to hold the presidential office without being elected as either president or vice-president . The choice of Ford to fulfill Spiro Agnew
's vacated role as vice president was based on Ford 's reputation for openness and honesty . `` In all the years I sat in the House , I never knew Mr. Ford to
make a dishonest statement nor a statement part - true and part - false . He never attempted to shade a statement , and I never heard him utter an unkind word ,
'' said Martha Griffiths . The trust the American people had in him was rapidly and severely tarnished by his pardon of Nixon . Nonetheless , many grant in
hindsight that he had respectably discharged with considerable dignity a great responsibility that he had not sought . His subsequent loss to Carter in 1976 has
come to be seen as an honorable sacrifice he made for the nation . In spite of his athletic record and remarkable career accomplishments , Ford acquired a
reputation as a clumsy , likable , and simple - minded Everyman .
</item 8>

<item 9>
President of the United States
Jump up ^ `` Obama signs protection bill for former presidents '' . The Washington Times . January 10 , 2013 . Retrieved August 14 , 2013 . Jump up ^ `` Shock
and Anger Flash Throughout the United States '' . Associated Press . March 31 , 1981 . Retrieved March 11 , 2011 . Jump up ^ `` Four Presidents '' . Reagan
Presidential Library , National Archives and Records Administration . Retrieved April 3 , 2011 . Jump up ^ `` Biography of Richard M. Nixon '' .
obamawhitehouse.archives.gov . , The White House . Jump up ^ 44 U.S.C. § 2112 Further reading Balogh , Brian and Bruce J. Schulman , eds . Recapturing the Oval
Office : New Historical Approaches to the American Presidency ( Cornell University Press , 2015 ) , 311 pp . Bumiller , Elisabeth ( January 2009 ) . `` Inside
the Presidency '' . National Geographic . 215 ( 1 ) : 130 -- 149 . Couch , Ernie .
</item 9>

<item 10>
Gerald Ford
An incident in 1975 , when he tripped while exiting Air Force One in Austria , was famously and repeatedly parodied by Chevy Chase , cementing Ford 's image as
a klutz . Pieces of Ford 's common Everyman image have also been attributed to Ford 's inevitable comparison to Nixon , as well as his perceived Midwestern
stodginess and self - deprecation . Civic and fraternal organizations ( edit ) Ford was a member of several civic organizations , including the Junior Chamber
of Commerce ( Jaycees ) , American Legion , AMVETS , Benevolent and Protective Order of Elks , Sons of the Revolution , and Veterans of Foreign Wars .
Freemasonry ( edit ) Ford was initiated into Freemasonry on September 30 , 1949 . He later said in 1975 , `` When I took my obligation as a master mason --
incidentally , with my three younger brothers -- I recalled the value my own father attached to that order . But I had no idea that I would ever be added to the
company of the Father of our Country and 12 other members of the order who also served as Presidents of the United States . ''
</item 10>
=== END ITEM id=5344f31de20a3e29 ===

=== ITEM id=d9ffc475e5784b73 ===
Gold sentences and facts:
Sentence 0: The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right : Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian Stewart Dick Taylor Bill Wyman Mick Taylor
  Fact 0.0: The Rolling Stones performed at Summerfest in Milwaukee in 2015.
  Fact 0.1: The photo shows, from left to right, Charlie Watts, Ronnie Wood, Mick Jagger and Keith Richards.
  Fact 0.2: The Rolling Stones have origin London, England.
  Fact 0.3: The Rolling Stones' genres are rock and blues.
  Fact 0.4: The Rolling Stones have been active from 1962 to the present.
  Fact 0.5: The Rolling Stones' labels include Decca, London, Rolling Stones, Virgin, ABKCO, Interscope, and Polydor.
  Fact 0.6: The Rolling Stones' associated acts include The New Barbarians, Billy Preston, Faces, The Pretty Things, John Mayall & the Bluesbreakers, and Bill Wyman's Rhythm Kings.
  Fact 0.7: The Rolling Stones' website is rollingstones.com.
  Fact 0.8: The current members of the Rolling Stones are Mick Jagger, Keith Richards, Charlie Watts, and Ronnie Wood.
  Fact 0.9: The past members of the Rolling Stones include Brian Jones, Ian Stewart, Dick Taylor, Bill Wyman, and Mick Taylor.
Sentence 1: Mick Jagger -- vocals , harmonica , rhythm guitar , percussion , keyboards ( 1962 -- present ) Keith Richards -- rhythm and lead guitar , bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing vocals ( 1975 -- present )
  Fact 1.0: Mick Jagger plays vocals, harmonica, rhythm guitar, percussion, and keyboards, and has been active from 1962 to the present.
  Fact 1.1: Keith Richards plays rhythm and lead guitar, bass guitar, and vocals, and has been active from 1962 to the present.
  Fact 1.2: Charlie Watts plays drums and percussion, and has been active from 1963 to the present.
  Fact 1.3: Ronnie Wood plays lead and rhythm guitar, bass guitar, and backing vocals, and has been active from 1975 to the present.
Sentence 2: The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums ) , and Ian Stewart ( piano ) . Stewart was removed from the official line - up in 1963 but continued as a touring member until his death in 1985 . Jones left the band less than a month before his death in 1969 , having already been replaced by Mick Taylor , who remained until 1974 . After Taylor left the band , Ronnie Wood took his place in 1975 and has been on guitar in tandem with Richards ever since . Following Wyman 's departure in 1993 , Darryl Jones joined as their touring bassist . Touring keyboardists for the band have been Nicky Hopkins ( 1967 -- 1982 ) , Ian McLagan ( 1978 -- 1981 ) , Billy Preston ( through the mid-1970s ) and Chuck Leavell ( 1982 -- present ) . The band was first led by Brian Jones , but after developing into the band 's songwriters , Jagger and Richards assumed leadership when Jones was dealing with personal troubles and legal issues .
  Fact 2.0: The Rolling Stones are an English rock band.
  Fact 2.1: The Rolling Stones were formed in London, England in 1962.
  Fact 2.2: The first stable line-up of the Rolling Stones consisted of Brian Jones, Mick Jagger, Keith Richards, Bill Wyman, Charlie Watts, and Ian Stewart.
  Fact 2.3: Brian Jones played guitar and harmonica in the first stable line-up.
  Fact 2.4: Mick Jagger was the lead vocalist in the first stable line-up.
  Fact 2.5: Keith Richards played guitar and backing vocals in the first stable line-up.
  Fact 2.6: Bill Wyman played bass in the first stable line-up.
  Fact 2.7: Charlie Watts played drums in the first stable line-up.
  Fact 2.8: Ian Stewart played piano in the first stable line-up.
  Fact 2.9: Ian Stewart was removed from the official line-up in 1963.
  Fact 2.10: Ian Stewart continued as a touring member of the Rolling Stones until his death in 1985.
  Fact 2.11: Brian Jones left the Rolling Stones less than a month before his death in 1969.
  Fact 2.12: Mick Taylor replaced Brian Jones in the band.
  Fact 2.13: Mick Taylor remained in the band until 1974.
  Fact 2.14: Ronnie Wood took Mick Taylor's place in 1975 after Taylor left the band.
  Fact 2.15: Ronnie Wood has been on guitar in tandem with Keith Richards ever since 1975.
  Fact 2.16: Darryl Jones joined the Rolling Stones as their touring bassist after Bill Wyman's departure in 1993.
  Fact 2.17: Bill Wyman departed from the Rolling Stones in 1993.
  Fact 2.18: Nicky Hopkins was a touring keyboardist for the Rolling Stones from 1967 to 1982.
  Fact 2.19: Ian McLagan was a touring keyboardist for the Rolling Stones from 1978 to 1981.
  Fact 2.20: Billy Preston was a touring keyboardist for the Rolling Stones through the mid-1970s.
  Fact 2.21: Chuck Leavell has been a touring keyboardist for the Rolling Stones since 1982.
  Fact 2.22: The band was first led by Brian Jones.
  Fact 2.23: Jagger and Richards became the band's songwriters.
  Fact 2.24: Jagger and Richards assumed leadership of the band when Brian Jones was dealing with personal troubles and legal issues.

Retrieved items:
<item 1>
The Rolling Stones
ISBN 978 - 1 - 59240 - 655 - 5 . External links ( edit ) Find more aboutThe Rolling Stonesat Wikipedia 's sister projects Media from Wikimedia Commons
Quotations from Wikiquote Data from Wikidata Discussion from Meta - Wiki Official website The Rolling Stones at Encyclopædia Britannica The Rolling Stones at
Curlie ( based on DMOZ ) `` The Rolling Stones '' . Rock and Roll Hall of Fame . The Rolling Stones at AllMusic The Rolling Stones discography at Discogs The
Rolling Stones on IMDb The Rolling Stones discography at MusicBrainz The Rolling Stones The Rolling Stones Mick Jagger Keith Richards Charlie Watts Ronnie Wood
Brian Jones Ian Stewart Mick Taylor Bill Wyman Tony Chapman Ricky Fenson Colin Golding Carlo Little Dick Taylor Video releases Let 's Spend the Night Together (
1983 ) Stones at the Max ( 1992 ) The Rolling Stones : Voodoo Lounge Live ( 1995 ) The Rolling Stones Rock and Roll Circus ( 1996 ) Bridges to Babylon Tour ' 97
-- 98 ( 1998 ) Four Flicks ( 2003 ) The Biggest Bang ( 2007 ) Ladies and Gentlemen : The Rolling Stones ( 2010
</item 1>

<item 2>
The Rolling Stones
The Rolling Stones - wikipedia The Rolling Stones Jump to : navigation , search `` Rolling Stones '' redirects here . For other uses , see Rolling Stones (
disambiguation ) . For the magazine , see Rolling Stone . The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right :
Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present
Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the
Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian
Stewart Dick Taylor Bill Wyman Mick Taylor The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted
of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums )
, and Ian Stewart ( piano ) .
</item 2>

<item 3>
The Rolling Stones
According to Richards , Jones named the band during a phone call to Jazz News . When asked by a journalist for the band 's name , Jones saw a Muddy Waters LP
lying on the floor ; one of the tracks was `` Rollin ' Stone '' . 1962 -- 1964 : building a following ( edit ) The back room of what was the Crawdaddy Club in
Richmond , London where the Rolling Stones had their first residency in 1963 Jones , Jagger , Richards , Stewart and Taylor played a gig billed as `` the Rollin
' Stones '' on 12 July 1962 , at the Marquee Club in London . Shortly afterwards the band went on their first tour of the UK , which they called a `` training
ground '' tour , because it was a new experience for all of them . Their material included the Chicago blues as well as Chuck Berry and Bo Diddley songs . At
that time , the line - up included neither bassist Bill Wyman , who joined in December 1962 , nor drummer Charlie Watts , who joined in January 1963 , thus
completing the band 's original rhythm section .
</item 3>

<item 4>
The Rolling Stones
( 1972 ) Goats Head Soup ( 1973 ) It 's Only Rock ' n Roll ( 1974 ) Black and Blue ( 1976 ) Some Girls ( 1978 ) Emotional Rescue ( 1980 ) Tattoo You ( 1981 )
Undercover ( 1983 ) Dirty Work ( 1986 ) Steel Wheels ( 1989 ) Voodoo Lounge ( 1994 ) Bridges to Babylon ( 1997 ) A Bigger Bang ( 2005 ) Blue & Lonesome ( 2016 )
See also ( edit ) The Rolling Stones portal Book : The Rolling Stones The Rolling Stones Museum List of highest - grossing concert tours Notes ( edit ) Jump up
^ Mick Avory himself has categorically denied `` on many occasions '' that he played with the Rollin ' Stones that night . In fact he only rehearsed twice with
them in the Bricklayers Arms pub , before they became known as the Rollin ' Stones . Jump up ^ Wyman 's book Rolling With The Stones incorrectly states the band
played the Alcove club that night . Jump up ^ The comma in the title of the song was later dropped .
</item 4>

<item 5>
The Rolling Stones
Hail ! Rock ' n ' Roll Wingless Angels Bill Wyman Studio albums Monkey Grip Green Ice Back to Basics Songs `` In Another Land '' `` ( Si Si ) Je Suis un Rock
Star '' VIAF : 156609707 LCCN : n80123108 ISNI : 0000 0001 2192 4040 GND : 2074630 - 1 SELIBR : 333421 SUDOC : 029185149 BNF : cb13906219j ( data ) BIBSYS :
11053891 MusicBrainz : b071f9fa - 14b0 - 4217 - 8e97 - eb41da73f598 NLA : 35460187 NDL : 00629452 NKC : ko2002161839 BNE : XX98942 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Rolling_Stones&oldid=837955424 '' Categories : The Rolling Stones 1962 establishments in England Atlantic Records
artists British Invasion artists British rhythm and blues boom musicians Columbia Records artists Counterculture of the 1960s Decca Records artists English
blues rock musical groups English
</item 5>

<item 6>
The Rolling Stones
4 1968 -- 1972 : `` Back to basics '' 1.5 1972 -- 1977 : Critical fluctuations and Ronnie Wood 1.6 1978 -- 1982 : Commercial peak 1.7 1983 -- 1988 : Band
turmoil and solo efforts 1.8 1989 -- 1999 : Comeback , return to popularity , and record - breaking tours 1.9 2000 -- 2011 : A Bigger Bang and continued success
1.10 2012 -- present : 50th anniversary and covers album 2 Musical development 3 Legacy 4 Tours 5 Band members 6 Discography 7 See also 8 Notes 9 References 9.1
Footnotes 9.2 Sources 10 Further reading 11 External links History ( edit ) Early History ( edit ) Keith Richards and Mick Jagger became childhood friends and
classmates in 1950 in Dartford , Kent , before the Jagger family moved to Wilmington , five miles ( 8.05 km ) away , in 1954 .
</item 6>

<item 7>
The Rolling Stones
They are also credited with changing the `` whole business model of popular music . '' The band has received -- and been nominated for -- multiple awards during
their 55 years as a band ; including three Grammy awards ( and 12 nominations ) , the Juno award for International Entertainer of the Year in 1991 , U.K. 's
Jazz FM Awards Album of the Year ( 2017 ) for their album Blue & Lonesome , and NME awards such as best live band and the NME award for best music film , for
their documentary Crossfire Hurricane . Tours ( edit ) Main article : The Rolling Stones concerts The Rolling Stones ' first concert was on 12 July 1962 at the
Marquee Club in London . The most documented of all the band 's concerts was the Altamont Free Concert at the Altamont Speedway in 1969 . For this concert , the
biker gang Hells Angels provided security , which resulted in a fan , Meredith Hunter , being stabbed and beaten to death by the Angels . Part of the tour and
the Altamont concert were documented in Albert and David Maysles ' film Gimme Shelter .
</item 7>

<item 8>
The Rolling Stones
bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing
vocals ( 1975 -- present ) Former members Brian Jones -- rhythm guitar , harmonica , keyboards , piano , backing vocals ( 1962 -- 1969 ; died 1969 ) Ian Stewart
-- keyboards , piano ( 1962 -- 1963 ; touring 1969 , 1975 -- 1976 , 1978 , 1981 -- 1982 ; died 1985 ) Mick Taylor -- lead guitar , bass guitar , backing vocals
( 1969 -- 1974 ; guest 1981 , 2012 -- 2014 ) Bill Wyman -- bass guitar , keyboards , piano , backing vocals ( 1962 -- 1993 ;
</item 8>

<item 9>
The Rolling Stones
keyboards , piano , backing vocals ( 1962 -- 1993 ; guest 2012 ) Early members Dick Taylor -- bass guitar ( 1962 ) Ricky Fenson -- bass guitar ( 1962 -- 1963 )
Colin Golding -- bass guitar ( 1962 -- 1963 ) Tony Chapman -- drums ( 1962 -- 1963 ) Carlo Little -- drums ( 1962 -- 1963 ) Current touring members Chuck
Leavell -- keyboards , backing vocals ( 1982 -- present ) Bernard Fowler -- backing vocals , percussion ( 1989 -- present ) Darryl Jones -- bass guitar ,
backing vocals ( 1993 -- present ) Matt Clifford -- keyboards , French horn , musical integrator ( 1989 -- 1990 , 2012 -- present ) Tim Ries -- saxophone ,
keyboards ( 1999 -- present ) Karl Denson -- saxophone ( 2014 -- present ) Sasha Allen -- backing vocals , co-lead vocals on `` Gimme Shelter '' ( 2016 --
present ) Former touring musicians Bobby Keys -- saxophone ( 1969 -- 1973 ; 1981 -- 2014 ;
</item 9>

<item 10>
The Rolling Stones
By 1963 they were finding their musical stride as well as popularity , and in 1964 two unscientific opinion polls rated them as Britain 's most popular group ,
even outranking the Beatles . The name of the band was changed shortly after their first gig to `` The Rolling Stones '' . The group 's then acting manager
Giorgio Gomelsky secured a Sunday afternoon residency at the Crawdaddy Club in Richmond in February 1963 , which , Gomelsky claimed , triggered an ``
international renaissance for the blues '' . In May 1963 , The Rolling Stones signed Andrew Loog Oldham as their manager ; he was a former publicist who had
been directed to the band by his previous clients , the Beatles . Because Oldham was only nineteen and had not reached the age of majority -- he was also
younger than anyone in the band -- he could not get an agent 's licence nor sign any contracts without his mother also signing . By necessity he joined with
booking agent Eric Easton in order to secure record financing and assistance booking venues . Gomelsky , who had no written agreement with the band , was not
consulted .
</item 10>
=== END ITEM id=d9ffc475e5784b73 ===
