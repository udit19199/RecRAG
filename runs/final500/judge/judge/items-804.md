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

=== ITEM id=7d9762a3aa13da58 ===
Gold sentences and facts:
Sentence 0: Because the states were preexisting political entities , the U.S. Constitution did not need to define or explain federalism in any one section but it often mentions the rights and responsibilities of state governments and state officials in relation to the federal government . The federal government has certain express powers ( also called enumerated powers ) which are powers spelled out in the Constitution , including the right to levy taxes , declare war , and regulate interstate and foreign commerce . In addition , the Necessary and Proper Clause gives the federal government the implied power to pass any law `` necessary and proper '' for the execution of its express powers . Other powers -- the reserved powers -- are reserved to the people or the states . The power delegated to the federal government was significantly expanded by the Supreme Court decision in McCulloch v. Maryland ( 1819 ) , amendments to the Constitution following the Civil War , and by some later amendments -- as well as the overall claim of the Civil War , that the states were legally subject to the final dictates of the federal government .
  Fact 0.0: The states were preexisting political entities.
  Fact 0.1: The U.S. Constitution did not need to define or explain federalism in any one section, because the states were preexisting political entities.
  Fact 0.2: The U.S. Constitution often mentions the rights and responsibilities of state governments and state officials in relation to the federal government.
  Fact 0.3: The federal government has certain express powers, also called enumerated powers.
  Fact 0.4: Express powers are powers spelled out in the Constitution.
  Fact 0.5: The federal government's express powers include the right to levy taxes.
  Fact 0.6: The federal government's express powers include the right to declare war.
  Fact 0.7: The federal government's express powers include the right to regulate interstate commerce.
  Fact 0.8: The federal government's express powers include the right to regulate foreign commerce.
  Fact 0.9: The Necessary and Proper Clause gives the federal government the implied power to pass any law 'necessary and proper' for the execution of its express powers.
  Fact 0.10: Other powers, the reserved powers, are reserved to the people or the states.
  Fact 0.11: The Supreme Court decision in McCulloch v. Maryland (1819) significantly expanded the power delegated to the federal government.
  Fact 0.12: Amendments to the Constitution following the Civil War significantly expanded the power delegated to the federal government.
  Fact 0.13: Some later amendments to the Constitution significantly expanded the power delegated to the federal government.
  Fact 0.14: The overall claim of the Civil War, that the states were legally subject to the final dictates of the federal government, significantly expanded the power delegated to the federal government.

Retrieved items:
<item 1>
Federalism
- J. Proudhon , The Principle of Federation , 1863 . A Comparative Bibliography : Regulatory Competition on Corporate Law A Rhetoric for Ratification : The
Argument of the Federalist and its Impact on Constitutional Interpretation Brainstorming National ( in French ) Teaching about Federalism in the United States -
From the Education Resources Information Center Clearinghouse for Social Studies / Social Science Education Bloomington , Indiana . An Ottawa , Ontario , Canada
- based international organization for federal countries that share best practices among countries with that system of government Tenth Amendment Center
Federalism and States Rights in the U.S. BackStory Radio episode on the origins and current status of Federalism Constitutional law scholar Hester Lessard
discusses Vancouver 's Downtown Eastside and jurisdictional justice McGill University ,
</item 1>

<item 2>
Federalism
The U.S. Constitution was written as a reaction to the Articles of Confederation , under which the United States was a loose confederation with a weak central
government . In contrast , Europe has a greater history of unitary states than North America , thus European `` federalism '' argues for a weaker central
government , relative to a unitary state . The modern American usage of the word is much closer to the European sense . As the power of the Federal government
has increased , some people have perceived a much more unitary state than they believe the Founding Fathers intended . Most people politically advocating ``
federalism '' in the United States argue in favor of limiting the powers of the federal government , especially the judiciary ( see Federalist Society , New
Federalism ) . In Canada , federalism typically implies opposition to sovereigntist movements ( most commonly Quebec separatism ) . The governments of Argentina
, Australia , Brazil , India , and Mexico , among others , are also organized along federalist principles . Federalism may encompass as few as two or three
internal divisions , as is the case in Belgium or Bosnia and Herzegovina .
</item 2>

<item 3>
Federalism
they are already constitutional , parliamentary , and administratively modernized states ) . European vs. American federalism ( edit ) Main articles : Federal
Europe and Federalism in the United States In Europe , `` Federalist '' is sometimes used to describe those who favor a common federal government , with
distributed power at regional , national and supranational levels . Most European federalists want this development to continue within the European Union .
European federalism originated in post-war Europe ; one of the more important initiatives was Winston Churchill 's speech in Zürich in 1946 . In the United
States , federalism originally referred to belief in a stronger central government . When the U.S. Constitution was being drafted , the Federalist Party
supported a stronger central government , while `` Anti-Federalists '' wanted a weaker central government . This is very different from the modern usage of ``
federalism '' in Europe and the United States . The distinction stems from the fact that `` federalism '' is situated in the middle of the political spectrum
between a confederacy and a unitary state . The U.S.
</item 3>

<item 4>
Federalism
Jump up ^ See Law , John ( 2013 ) , p. 104 . http://www.on-federalism.eu/attachments/169_download.pdf This author identifies two distinct federal forms , where
before only one was known , based upon whether sovereignty ( conceived in its core meaning of ultimate authority ) resides in the whole ( in one people ) or in
the parts ( in many peoples ) . This is determined by the absence or presence of a right of secession for the parts . The structures are termed , respectively ,
the federal state ( or federation ) and the federal union of states ( or federal union ) . Jump up ^ Madison , James , Hamilton , Alexander and Jay , John (
1987 ) The Federalist Papers , Penguin , Harmondsworth , p. 259 . Jump up ^ Law , John ( 2012 ) ' Sense on Federalism ' , in Political Quarterly , Vol. 83 , No.
3 , p. 544 .
</item 4>

<item 5>
Federalism
See also ( edit ) Consociationalism Cooperative federalism Democratic World Federalists Federal Union Layer cake federalism Pillarisation States ' rights Union
of Utrecht World Federalist Movement Notes and references ( edit ) Jump up ^ In 1946 , scholar Kenneth Wheare observed that the two levels of government in the
US were ' co-equally supreme ' . In this he echoed the perspective of the founding fathers , James Madison in Federalist 39 having seen the several states as
forming ' distinct and independent portions of the supremacy ' in relation to the general government . Wheare , Kenneth ( 1946 ) Federal Government , Oxford
University Press , London , pp. 10 - 15 . Madison , James , Hamilton , Alexander and Jay , John ( 1987 ) The Federalist Papers , Penguin , Harmondsworth , p.
258 . Jump up ^ Law , John ( 2013 ) ' How Can We Define Federalism ? ' , in Perspectives on Federalism , Vol. 5 , No. 3 , pp. E105 - 6 .
</item 5>

<item 6>
Federalism
American government has evolved from a system of dual federalism to one of associative federalism . In `` Federalist No. 46 , '' James Madison asserted that the
states and national government `` are in fact but different agents and trustees of the people , constituted with different powers . '' Alexander Hamilton ,
writing in `` Federalist No. 28 , '' suggested that both levels of government would exercise authority to the citizens ' benefit : `` If their ( the peoples ' )
rights are invaded by either , they can make use of the other as the instrument of redress . '' ( 1 ) The United States is composed of fifty self - governing
states and several territories . Because the states were preexisting political entities , the U.S. Constitution did not need to define or explain federalism in
any one section but it often mentions the rights and responsibilities of state governments and state officials in relation to the federal government . The
federal government has certain express powers ( also called enumerated powers ) which are powers spelled out in the Constitution , including the right to levy
taxes , declare war , and regulate interstate and foreign commerce .
</item 6>

<item 7>
Federalism divides authority between national and regional governments, contrasting with unitary states. In the U.S., it arose in response to the weak central
government under the Articles of Confederation. American federalism evolved from dual to associative federalism; Madison described both levels as agents of the
people, while Hamilton saw each as a safeguard against abuses by the other. The Constitution grants the federal government enumerated powers, including
taxation, declaring war, and regulating interstate and foreign commerce. Federalism also shapes politics in countries
</item 7>

<item 8>
Federalism divides powers between two coequal levels of government, exemplified by the United States under its 1787 Constitution. Unlike confederalism, where
regional governments dominate, or devolution in a unitary state, where the central government dominates, federalism occupies a middle position between them.
Examples include the United States, Canada, Brazil, Germany, Switzerland, Argentina, Australia, and India. The European Union is sometimes described as a
pioneering federal union of states. The article also surveys federalism
</item 8>

<item 9>
Federalism
This was reaffirmed by the Tenth Amendment to the United States Constitution , which reserves all powers and rights that are not delegated to the Federal
Government as left to the States and to the people . Bicameralism ( edit ) The structures of most federal governments incorporate mechanisms to protect the
rights of component states . One method , known as ' intrastate federalism ' , is to directly represent the governments of component states in federal political
institutions . Where a federation has a bicameral legislature the upper house is often used to represent the component states while the lower house represents
the people of the nation as a whole . A federal upper house may be based on a special scheme of apportionment , as is the case in the senates of the United
States and Australia , where each state is represented by an equal number of senators irrespective of the size of its population .
</item 9>

<item 10>
Federalism
Its distinctive feature , exemplified in the founding example of modern federalism by the United States of America under the Constitution of 1787 , is a
relationship of parity between the two levels of government established . It can thus be defined as a form of government in which there is a division of powers
between two levels of government of equal status . Federalism differs from confederalism , in which the general level of government is subordinate to the
regional level , and from devolution within a unitary state , in which the regional level of government is subordinate to the general level . It represents the
central form in the pathway of regional integration or separation , bounded on the less integrated side by confederalism and on the more integrated side by
devolution within a unitary state . Leading examples of the federation or federal state include the United States , Canada , Brazil , Germany , Switzerland ,
Argentina , Australia and India . Some also today characterize the European Union as the pioneering example of federalism in a multi-state setting , in a
concept termed the federal union of states . Contents ( hide ) 1 Overview 1.1 Explanations for adoption of federalist systems 2 European vs.
</item 10>
=== END ITEM id=7d9762a3aa13da58 ===

=== ITEM id=8171336066b7fd1c ===
Gold sentences and facts:
Sentence 0: The Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take ) retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority ; or a substantial and specific danger to public health or safety .
  Fact 0.0: The Whistleblower Protection Act of 1989 is Pub. L. 101-12, as amended.
  Fact 0.1: The Whistleblower Protection Act of 1989 is a United States federal law.
  Fact 0.2: The Whistleblower Protection Act protects federal whistleblowers who work for the government and report agency misconduct.
  Fact 0.3: A federal agency violates the Whistleblower Protection Act if agency authorities take, or threaten to take, retaliatory personnel action against any employee or applicant.
  Fact 0.4: The retaliatory personnel action must be because of disclosure of information by that employee or applicant.
  Fact 0.5: Whistleblowers may file complaints that they believe reasonably evidence a violation of a law, rule or regulation.
  Fact 0.6: Whistleblowers may file complaints that they believe reasonably evidence gross mismanagement.
  Fact 0.7: Whistleblowers may file complaints that they believe reasonably evidence gross waste of funds.
  Fact 0.8: Whistleblowers may file complaints that they believe reasonably evidence an abuse of authority.
  Fact 0.9: Whistleblowers may file complaints that they believe reasonably evidence a substantial and specific danger to public health or safety.

Retrieved items:
<item 1>
Whistleblower Protection Act
Legal cases ( edit ) The U.S. Supreme Court , in the case of Garcetti v. Ceballos , 04 - 473 , ruled in 2006 that government employees do not have protection
from retaliation by their employers under the First Amendment of the Constitution when they speak pursuant to their official job duties . The U.S. Merit Systems
Protection Board ( MSPB ) uses agency lawyers in the place of administrative law judges to decide federal employees ' whistleblower appeals . These lawyers ,
dubbed `` attorney examiners , '' deny 98 % of whistleblower appeals ; the Board and the Federal Circuit Court of Appeals give great deference to their initial
decisions , resulting in affirmance rates of 97 % and 98 % , respectively .
</item 1>

<item 2>
Whistleblower Protection Act
November 7 , 2014 . Retrieved April 27 , 2016 . Jump up ^ https://obamawhitehouse.archives.gov/sites/default/files/image/ppd-19.pdf Jump up ^ WHISTLEBLOWER
PROTECTION ENHANCEMENT ACT OF 2012 REPORT OF THE COMMITTEE ON HOMELAND SECURITY AND GOVERNMENTAL AFFAIRS , UNITED STATES SENATE TO ACCOMPANY S. 743 . April 19 ,
2012 Jump up ^ https://www.gpo.gov/fdsys/pkg/CRPT-112srpt155/pdf/CRPT-112srpt155.pdf Jump up ^ `` H.R. 4197 - Summary '' . United States Congress . Retrieved
July 15 , 2014 . Jump up ^ `` H.R. 4197 - All Actions '' . United States Congress . Retrieved July 15 , 2014 . Jump up ^ Joe Davidson . How FBI blocks
whistleblower fighting dismissal ; new bill could help others . Washington Post 2 Feb 2016 .
</item 2>

<item 3>
Whistleblower Protection Act
Specifically , the Federal Circuit has wrongly accorded a narrow definition to the type of disclosure that qualifies for whistleblower protection . Additionally
, the lack of remedies under current law for most whistleblowers in the intelligence community and for whistleblowers who face retaliation in the form of
withdrawal of the employee 's security clearance leaves unprotected those who are in a position to disclose wrongdoing that directly affects our national
security . '' S. 743 would address these problems by restoring the original congressional intent of the WPA to adequately protect whistleblowers , by
strengthening the WPA , and by creating new whistleblower protections for intelligence employees and new protections for employees whose security clearance is
withdrawn in retaliation for having made legitimate whistleblower disclosures . S. 743 ultimately became Pub. L. 112 - 199 ( S. Rep. 112 - 155 ) . Related
legislation ( edit ) This section needs to be updated . Please update this article to reflect recent events or newly available information . ( January 2017 ) On
July 14 , 2014 , the United States House of Representatives voted to pass the All Circuit Review Extension Act ( H.R.
</item 3>

<item 4>
Whistleblower Protection Act
4197 ; 113th Congress ) , a bill that gives authority to federal employees who want to appeal their judgment to any federal court , and which allows
whistleblowers to appeal to any U.S. Court of Appeals that has jurisdiction . The bill would extend from three years after the effective date of the
Whistleblower Protection Enhancement Act of 2012 ( i.e. , December 27 , 2012 ) , the period allowed for : ( 1 ) filing a petition for judicial review of Merit
Systems Protection Board decisions in whistleblower cases , and ( 2 ) any review of such a decision by the Director of the Office of Personnel Management ( OPM
) . There is also a bill in its early stages called the FBI Whistleblower Protection Enhancement Act . This bill was introduced by Senators Grassley and Leahy
in 2015 . It would protect whistleblowers in the FBI more completely from reprisals by supervisors , by assigning administrative law judges to adjudicate the
cases , and enabling FBI employees to appeal their decisions to the courts . This bill is still in the process of possibly getting passed and has individuals
cosponsoring it in March 2016 .
</item 4>

<item 5>
Whistleblower Protection Act
The most common characteristics for a court claim that are encompassed within the protection of the Act include : that the plaintiff is an employee or person
covered under the specific statutory or common law relied upon for action , that the defendant is an employer or person covered under the specific statutory or
common law relied upon for the action , that the plaintiff engaged in protected whistleblower activity , that the defendant knew or had knowledge that the
plaintiff engaged in such activity , that there was retaliatory action taken against the one doing the whistleblowing and that the unfair treatment would not
have occurred if the plaintiff had n't brought to attention the activities . Robert MacLean blew the whistle on the fact that the TSA had cut its funding for
more air marshals . In 2009 MacLean , represented by the Government Accountability Project , challenged his dismissal at the Merit Systems Protection Board , on
the grounds that `` his disclosure of the text message was protected under the Whistleblower Protection Act of 1989 , because he ' reasonably believe ( d ) '
that the leaked information disclosed ' a substantial and specific danger to public health or safety ' . ''
</item 5>

<item 6>
Whistleblower Protection Act
Whistleblower Protection Act - Wikipedia Whistleblower Protection Act Jump to : navigation , search For the Act in India , see Whistle Blowers Protection Act ,
2011 . Whistleblower Protection Act Long title An Act to amend title 5 , United States Code , to strengthen the protections available to Federal employees
against prohibited personnel practices , and for other purposes . Nicknames Whistleblower Protection Act of 1989 Enacted by the 101st United States Congress
Effective April 10 , 1989 Citations Public law 101 - 12 Statutes at Large 103 Stat. 16 Codification Titles amended 5 U.S.C. : Government Organization and
Employees U.S.C. sections amended 5 U.S.C. ch. 12 § 1201 et seq . Legislative history Introduced in the Senate as S.
</item 6>

<item 7>
Whistleblower Protection Act
Legislative history Introduced in the Senate as S. 20 by Carl Levin ( D -- MI ) on January 25 , 1989 Passed the Senate on March 16 , 1989 ( 97 - 0 , Roll call
vote 24 , via Senate.gov ) Passed the House on March 21 , 1989 ( Agreed voice vote ) Signed into law by President George H.W. Bush on April 10 , 1989 The
Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the
government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take )
retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file
complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority
; or a substantial and specific danger to public health or safety .
</item 7>

<item 8>
Whistleblower Protection Act
Contents ( hide ) 1 Authorized Federal Agencies 2 Legal Cases 3 Whistleblower Protection Enhancement Act and Presidential Policy Directive 19 4 Related
legislation 5 See also 6 References 7 External links Authorized federal agencies ( edit ) The Office of Special Counsel investigates federal whistleblower
complaints . In October 2008 , then - special counsel Scott Bloch resigned amid an FBI investigation into whether he obstructed justice by illegally deleting
computer files following complaints that he had retaliated against employees who disagreed with his policies . Then - Senator Barack Obama made a campaign vow
to appoint a special counsel committed to whistleblower rights . It was not until April 2011 that President Obama 's appointee Carolyn Lerner was confirmed by
the Senate . Today , the primary mission of OSC is to safeguard the merit system by protecting federal employees and applicants from prohibited personnel
practices , especially reprisal for whistleblowing . The Merit Systems Protection Board , a quasi-judicial agency that adjudicates whistleblower complaints ,
uses appointed administrative law judges who often back the government . Since 2000 , the board has ruled for whistleblowers just three times in 56 cases
decided on their merits , according to a Government Accountability Project analysis .
</item 8>

<item 9>
Whistleblower Protection Act
See also ( edit ) Whistleblower protection in United States False Claims Act Federal crime Immunity from prosecution Informant List of whistleblowers Qui tam
Testimony Turn state 's evidence White collar crime Witness Witness intimidation United States Federal Witness Protection Program United States Marshals Service
References ( edit ) Jump up ^ Public Law 101 - 12 ( 1989 ) at gpo.gov Jump up ^ Peter Eisler ( 2010 - 03 - 15 ) . `` Whistleblowers '' . USA TODAY . Jump up ^
High Court Trims Whistleblower Rights Jump up ^ McCarthy , Robert J. `` Blowing in the Wind : Answers for Federal Whistleblowers '' . William & Mary Policy
Review 3 : 184 ( 2012 ) . Jump up ^ `` Know Your Rights FAQ '' . National Whistleblowers Center . Retrieved April 27 , 2016 . Jump up ^ `` Whistleblower
protection case reaches US Supreme Court '' . Courage Foundation . November 7 , 2014 . Retrieved April 27 , 2016 .
</item 9>

<item 10>
Whistleblower Protection Act
Obama appointed a new chairperson and vice chairperson with backgrounds as federal worker advocates , but Tom Devine of GAP says , `` It 's likely to take years
for them to turn things around . '' Currently , this office works to protect the Merit System Principles and promote an effective Federal workforce free of
Prohibited Personnel Practices . The Court of Appeals for the Federal Circuit was established under Article III of the Constitution on October 1 , 1982 . It is
the only court empowered to hear appeals of whistleblower cases decided by the merit board , has been criticized by Senator Grassley ( R - Iowa ) and others in
Congress for misinterpreting whistleblower laws and setting a precedent that is hostile to claimants . Between 1994 and 2010 , the court had ruled for
whistleblowers in only three of 203 cases decided on their merits , GAP 's analysis found . Legal cases ( edit ) The U.S.
</item 10>
=== END ITEM id=8171336066b7fd1c ===

=== ITEM id=521d7790f9ac0334 ===
Gold sentences and facts:
Sentence 0: The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert .
  Fact 0.0: The Second Great Awakening occurred in several episodes.
  Fact 0.1: The Second Great Awakening occurred over different denominations.
  Fact 0.2: The revivals of the Second Great Awakening were very similar to each other.
  Fact 0.3: Revival meetings were the most effective form of evangelizing during this period.
  Fact 0.4: Revival meetings cut across geographical boundaries.
  Fact 0.5: The revival movement quickly spread throughout Kentucky, Tennessee and southern Ohio.
  Fact 0.6: Each denomination had assets that allowed it to thrive on the frontier.
  Fact 0.7: The Methodists had an efficient organization.
  Fact 0.8: The Methodist organization depended on itinerant ministers known as circuit riders.
  Fact 0.9: Circuit riders sought out people in remote frontier locations.
  Fact 0.10: Circuit riders came from among the common people.
  Fact 0.11: Coming from among the common people helped circuit riders establish rapport with the frontier families they hoped to convert.
Sentence 1: The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ .
  Fact 1.0: The revivals enrolled millions of new members in existing evangelical denominations.
  Fact 1.1: The revivals led to the formation of new denominations.
  Fact 1.2: Many converts believed that the Awakening heralded a new millennial age.
  Fact 1.3: The Second Great Awakening stimulated the establishment of many reform movements.
  Fact 1.4: The reform movements were designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ.
Sentence 2: In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be saved was from grace from God .
  Fact 2.0: In the newly settled frontier regions, the revival was implemented through camp meetings.
  Fact 2.1: Camp meetings often provided the first encounter for some settlers with organized religion.
  Fact 2.2: Camp meetings were important as social venues.
  Fact 2.3: A camp meeting was a religious service of several days' length with preachers.
  Fact 2.4: Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship.
  Fact 2.5: Participating in a religious revival with crowds of hundreds and perhaps thousands of people was sheer exhilaration.
  Fact 2.6: That exhilaration inspired the dancing, shouting, and singing associated with camp meeting events.
  Fact 2.7: The revivals followed an arc of great emotional power.
  Fact 2.8: The revivals emphasized the individual's sins and need to turn to Christ.
  Fact 2.9: The revivals were restored by a sense of personal salvation.
  Fact 2.10: Upon their return home, most converts joined or created small local churches.
  Fact 2.11: The small local churches grew rapidly.
  Fact 2.12: The Second Great Awakening marked a religious transition in society in America.
  Fact 2.13: Many Americans from the Calvinist sect emphasized man's inability to save themselves.
  Fact 2.14: Many Americans from the Calvinist sect emphasized that their only way to be saved was by grace from God.

Retrieved items:
<item 1>
Second Great Awakening
a re-emphasis on Wesleyan teachings on sanctification emerged during the Second Great Awakening , leading to a distinction between Mainline Methodism and
Holiness churches . Restoration movement ( edit ) Main article : Restoration Movement See also : Restorationism The idea of restoring a `` primitive '' form of
Christianity grew in popularity in the U.S. after the American Revolution . This desire to restore a purer form of Christianity without an elaborate hierarchy
contributed to the development of many groups during the Second Great Awakening , including the Mormons , Baptists and Shakers . Several factors made the
restoration sentiment particularly appealing during this time period : To immigrants in the early 19th century , the land in the United States seemed pristine ,
edenic and undefiled -- `` the perfect place to recover pure , uncorrupted and original Christianity '' -- and the tradition - bound European churches seemed
out of place in this new setting . A primitive faith based on the Bible alone promised a way to sidestep the competing claims of the many denominations
available and for congregations to find assurance of being right without the security of an established national church . The Restoration Movement began during
, and was greatly influenced by , the Second Great Awakening .
</item 1>

<item 2>
Second Great Awakening
West ( edit ) Main article : Revival of 1800 In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided
the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several
days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of
participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these
events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of
personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a
religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be
saved was from grace from God . The Revival of 1800 in Logan County , Kentucky , began as a traditional Presbyterian sacramental occasion .
</item 2>

<item 3>
Second Great Awakening
They were committed to individuals ' achieving a personal relationship with Christ . Churches with roots in this movement include the Churches of Christ ,
Christian Church ( Disciples of Christ ) , and the Evangelical Christian Church in Canada Church membership soars ( edit ) 1839 Methodist camp meeting The
Methodist circuit riders and local Baptist preachers made enormous gains ; to a lesser extent the Presbyterians gained members , particularly with the
Cumberland Presbyterian Church in sparsely settled areas . As a result , the numerical strength of the Baptists and Methodists rose relative to that of the
denominations dominant in the colonial period -- the Anglicans , Presbyterians , Congregationalists . Among the new denominations that grew from the religious
ferment of the Second Great Awakening are the Churches of Christ , Christian Church ( Disciples of Christ ) , the Seventh - day Adventist Church , and the
Evangelical Christian Church in Canada . The converts during the Second Great Awakening were predominantly female . A 1932 source estimated at least three
female converts to every two male converts between 1798 and 1826 . Young people ( those under 25 ) also converted in greater numbers , and were the first to
convert .
</item 3>

<item 4>
Second Great Awakening
Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements
designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ . Historians named the Second Great Awakening in the context of the
First Great Awakening of the 1730s and 1740s and of the Third Great Awakening of the late 1850s to early 1900s . These revivals were part of a much larger
Romantic religious movement that was sweeping across Europe at the time , mainly throughout England , Scotland , and Germany .
</item 4>

<item 5>
Second Great Awakening
While Protestant religion had previously played an important role on the American political scene , the Second Great Awakening strengthened the role it would
play . See also ( edit ) Baptist portal Methodism portal Calvinism portal Seventh - day Adventist Church portal Latter Day Saints portal LDS Church portal
Advent Christian Church Christian revival Christianity in the 19th century The Church of Jesus Christ of Latter - day Saints Cumberland Presbyterian Church
Ethnocultural politics in the United States Holiness movement Restoration Movement Seventh - day Adventist Church References ( edit ) ^ Jump up to : Timothy L.
Smith , Revivalism and Social Reform : American Protestantism on the Eve of the Civil War ( 1957 ) . Jump up ^ Heyrman , Christine Leigh . `` The First Great
Awakening '' . Divining America , TeacherServe . National Humanities Center . Jump up ^ Henry B. Clark ( 1982 ) . Freedom of Religion in America : Historical
Roots , Philosophical Concepts , Contemporary Problems . Transaction Publishers . p. 16 . ^ Jump up to : Nancy Cott , `` Young Women in the Second Great
Awakening in New England , '' Feminist Studies , ( 1975 ) , 3 # 1 p.
</item 5>

<item 6>
Second Great Awakening
the Second reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skepticism , deism , and
rationalism left over from the Enlightenment . At about the same time , similar movements flourished in Europe . Pietism was sweeping German countries .
Evangelicalism was waxing strong in England . The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals
were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement
quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had
an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit
riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert . Theology ( edit ) Main article
: Postmillennialism Postmillennialism theology dominated American Protestantism in the first half of the 19th century .
</item 6>

<item 7>
Second Great Awakening
See Uncommon Christian Ministries Long , Kimberly Bracken . `` The Communion Sermons of James Mcgready : Sacramental Theology and Scots - Irish Piety on the
Kentucky Frontier '' , Journal of Presbyterian History , 2002 80 ( 1 ) : 3 -- 16 . ISSN 0022 - 3883 . JSTOR 23336302 . Loveland Anne C. Southern Evangelicals
and the Social Order , 1800 -- 1860 , ( 1980 ) McLoughlin William G. Modern Revivalism , 1959 . McLoughlin William G. Revivals , Awakenings , and Reform : An
Essay on Religion and Social Change in America , 1607 -- 1977 , 1978 . Marsden , George M. The Evangelical Mind and the New School Presbyterian Experience : A
Case Study of Thought and Theology in Nineteenth - Century America ( 1970 ) . Meyer , Neil . `` Falling for the Lord : Shame , Revivalism , and the Origins of
the Second Great Awakening . ''
</item 7>

<item 8>
Second Great Awakening
Second Great Awakening - wikipedia Second Great Awakening Jump to : navigation , search An engraving of a Methodist camp meeting in 1819 ( Library of Congress )
. Great Awakening First ( c. 1730 -- 1755 ) Second ( c. 1790 -- 1840 ) Third ( c. 1855 -- 1900 ) Fourth ( c. 1960 -- 1980 ) The Second Great Awakening was a
Protestant religious revival during the early 19th century in the United States . The movement began around 1790 , gained momentum by 1800 and , after 1820 ,
membership rose rapidly among Baptist and Methodist congregations whose preachers led the movement . It was past its peak by the late 1850s . The Second Great
Awakening reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skeptical rationalism and deism of
the Enlightenment . The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many
converts believed that the Awakening heralded a new millennial age .
</item 8>

<item 9>
Second Great Awakening
Contents ( hide ) 1 Spread of revivals 1.1 Background 1.2 Theology 1.3 Burned - over district 1.4 West and Tidewater South 1.5 West 1.6 Church membership soars
2 Subgroups 2.1 Adventism 2.2 Holiness movement 2.3 Restoration Movement 3 Culture and society 4 Slaves and free Africans 5 Women 6 Prominent figures 7
Political implications 8 See also 9 References 10 Further reading 10.1 Historiography Spread of revivals ( edit ) Part of a series on Baptists Background ( show
) Christianity Protestantism Puritanism Anabaptism Doctrine ( show ) Priesthood of all believers Individual soul liberty Separation of church and state Sola
scriptura Congregationalism Ordinances Offices Confessions Key figures ( show ) John Smyth Thomas Helwys Roger Williams John Clarke John Bunyan Shubal Stearns
Andrew Fuller Charles Spurgeon D.N. Jackson James Robinson Graves William Bullein Johnson William Carey Luther Rice Martin Luther King Jr .
</item 9>

<item 10>
Second Great Awakening
The Restoration Movement began during , and was greatly influenced by , the Second Great Awakening . While the leaders of one of the two primary groups making
up this movement , Thomas Campbell and Alexander Campbell , resisted what they saw as the spiritual manipulation of the camp meetings , the revivals contributed
to the development of the other major branch , led by Barton W. Stone . The Southern phase of the Awakening `` was an important matrix of Barton Stone 's reform
movement '' and shaped the evangelistic techniques used by both Stone and the Campbells . Culture and society ( edit ) Efforts to apply Christian teaching to
the resolution of social problems presaged the Social Gospel of the late 19th century . Converts were taught that to achieve salvation they needed not just to
repent personal sin but also work for the moral perfection of society , which meant eradicating sin in all its forms . Thus , evangelical converts were leading
figures in a variety of 19th century reform movements . Congregationalists set up missionary societies to evangelize the western territory of the northern tier
. Members of these groups acted as apostles for the faith , and also as educators and exponents of northeastern urban culture .
</item 10>
=== END ITEM id=521d7790f9ac0334 ===

=== ITEM id=253c0c9a7f06aa70 ===
Gold sentences and facts:
Sentence 0: On the November 19 , 1999 , episode of Millionaire , Carpenter proceeded to advance to the million - dollar question without using any lifelines . He then used his Phone - A-Friend to call his father not for help , but rather to tell him he was going to win the game . Carpenter answered the question correctly and became the show 's first millionaire . His win gave him national recognition and led to multiple talk show appearances , as well as reappearances on Millionaire itself .
  Fact 0.0: On the November 19, 1999 episode of Millionaire, Carpenter advanced to the million-dollar question.
  Fact 0.1: Carpenter advanced to the million-dollar question without using any lifelines.
  Fact 0.2: Carpenter used Phone-A-Friend to call his father.
  Fact 0.3: Carpenter did not call his father for help with the question.
  Fact 0.4: Carpenter called his father to tell him he was going to win the game.
  Fact 0.5: Carpenter answered the million-dollar question correctly.
  Fact 0.6: Carpenter became the show's first millionaire.
  Fact 0.7: Carpenter's win gave him national recognition.
  Fact 0.8: Carpenter's win led to multiple talk show appearances.
  Fact 0.9: Carpenter's win led to reappearances on Millionaire itself.

Retrieved items:
<item 1>
John Carpenter (game show contestant)
He later participated as part of the Mob on NBC 's 1 vs. 100 on October 27 , 2006 , and as a contestant on the Game Show Network game show Grand Slam .
Carpenter appeared on the August 16 , 2009 episode of Millionaire in prime time for its tenth anniversary . In the audience with him were his father , his wife
, and his son . Additionally , he was also the first expert in the `` Ask the Expert '' lifeline for the eighth season of the syndicated series . See also (
edit ) Biography portal List of Who Wants to Be a Millionaire ? top prize winners References ( edit ) Jump up ^ `` Millionaire Makeovers ! '' . People . Vol. 54
no . 20 . Time Inc . November 13 , 2000 . Archived from the original on September 11 , 2015 . Retrieved July 8 , 2015 . Jump up ^ Petrozello , Donna ( February
3 , 2000 ) . `` Million - plu $ reply an educated guess '' . New York Daily News . p. 101 .
</item 1>

<item 2>
John Carpenter (game show contestant)
, 15 Years After It Premiered '' . Huffington Post . Archived from the original on August 22 , 2015 . Retrieved July 9 , 2015 . Jump up ^ de Moraes , Lisa (
November 23 , 1999 ) . `` Drab ' Millionaire ' Winner Emerges as a Media Animal '' . The Washington Post . p . C07 . Jump up ^ Moore , Frazier ( January 6 ,
2001 ) . `` Wizard of ' Oz ' lives with permanent reminder of gritty show '' . The San Diego Union Tribune . p . E6 . Jump up ^ `` Matching Wits with the
Million - Dollar Mind : The World ; s Hardest Trivia Quizzes from America 's First Quiz Show Millionaire '' . The Book Depository . Archived from the original
on October 25 , 2015 . Retrieved October 24 , 2015 . Jump up ^ Deggans , Eric ( May 12 , 2000 ) . `` Miami ' Millionaire ' gets 2nd shot at final answer '' .
St.
</item 2>

<item 3>
John Carpenter (game show contestant)
Carpenter answered the question correctly and became the show 's first millionaire . His win gave him national recognition and led to multiple talk show
appearances , as well as reappearances on Millionaire itself . Contents ( hide ) 1 Career and family 2 Who Wants to Be a Millionaire 3 Later work 3.1 Other game
show appearances 4 See also 5 References 6 External links Career and family ( edit ) Carpenter is from Northampton , Massachusetts . His father , Tom , worked
as a computer program analyst for the Department of Veterans Affairs , while his mother , Gail , served as an administrative assistant for the Massachusetts
Audubon Society . In 1986 , he enrolled at Rutgers University and graduated in 1990 with a degree in economics . In January 1991 , he joined the Internal
Revenue Service after completing government exams and tests . In November 1996 , Carpenter met his future wife , Deborah , who was studying for a master 's
degree at Southern Connecticut State University and worked as a manager of a Fleet Bank branch in New Haven . They married in August 1998 .
</item 3>

<item 4>
John Carpenter (game show contestant)
The $1 million question was , `` Which of these U.S. Presidents appeared on the television series ' Laugh - In ' ? '' , with the choices being A ) Lyndon
Johnson , B) Richard Nixon , C ) Jimmy Carter , and D ) Gerald Ford . Carpenter used his Phone - A-Friend lifeline to call his father not for help , but rather
to inform him that he knew the answer . Carpenter later explained , `` I thought I 'd look so cocky if I did n't use any lifelines , so I faked it . '' With his
win , Carpenter became the first contestant in the worldwide Millionaire franchise to win the show 's top prize . He said that the only question that had
flustered him was one which asked for the location of the Gunfight at the O.K. Corral . Carpenter eventually remembered that the film Tombstone included the
gunfight , and he replied correctly with the answer ' Tombstone , Arizona ' . While taking a vacation after his win , Carpenter considered quitting his job with
the IRS , but eventually decided against it .
</item 4>

<item 5>
John Carpenter (game show contestant)
He explained to Kiplinger 's Personal Finance that `` after the taxes , it 's not change - your - life kind of money if you want to eat every day . '' Carpenter
also described the fame as having a bigger impact on his life than the money , later stating : `` The money does n't change your life . What happens afterwards
might . '' Later work ( edit ) Shortly after winning on Millionaire , Carpenter played himself in a Saturday Night Live skit . Donald Trump , played by Darrell
Hammond , announced that Carpenter would be his running - mate in the presidential election . Afterward , Carpenter pretended to call his father , then shouted
, `` Live from New York , it 's Saturday Night ! '' Carpenter also appeared on Good Morning America , Oprah Winfrey Show , Live with Regis and Kathie Lee , and
Late Show with David Letterman . Carpenter appeared as himself in the second half of the fourth season of Oz . He plays a contestant in a fictional TV game show
called Up Your Ante that the prisoners in Em City are watching .
</item 5>

<item 6>
John Carpenter (game show contestant)
`` Miami ' Millionaire ' gets 2nd shot at final answer '' . St. Petersburg Times . Archived from the original on April 5 , 2008 . Retrieved October 9 , 2014 .
Jump up ^ Bracht , Mel ( May 26 , 2000 ) . `` Money is n't the root of happiness : Tulsan has more fun in 2nd Millionaire gig '' . NewsOK . Oklahoma Publishing
Company . Archived from the original on March 13 , 2017 . Retrieved October 9 , 2014 . Jump up ^ Super Millionaire . Season 1 . Episode 2 . February 23 , 2004 .
ABC . Jump up ^ 1 vs. 100 . Season 1 . Episode 3 . October 27 , 2006 . NBC . Jump up ^ Owen , Rob ( August 2 , 2007 ) . `` Dennis Miller rallies around game
show all - stars '' . Pittsburgh Post-Gazette . p . WE - 33 .
</item 6>

<item 7>
John Carpenter (game show contestant)
They married in August 1998 . At the time of his appearance on Millionaire , he was 31 years old . When he revealed his profession as an IRS officer on
Millionaire , Carpenter was playfully booed by the audience . Who Wants to Be a Millionaire ( edit ) $1 million ( 15 of 15 ) - no time limit Which of these U.S.
Presidents appeared on the television series `` Laugh - In '' ? A : Lyndon Johnson B : Richard Nixon C : Jimmy Carter D: Gerald Ford Carpenter was originally
uninterested in Millionaire but eventually tuned in one night after dinner while having friends over at his house . When he found the show 's higher - tier
questions to not be , in his words , `` any more difficult '' than the lower - tier ones , he decided to call in to the show 's hotline for a chance to become a
contestant . Carpenter answered all of the hotline questions correctly and was on the show within two days . Host Regis Philbin described Carpenter as having ``
cruised right through those first fourteen questions , '' he had proceeded to reach the final question without using any of his lifelines .
</item 7>

<item 8>
John Carpenter (game show contestant)
The show within the show is hosted by Gordon Elliott , with Eartha Kitt and Didi Conn appearing as celebrity participants . With Rod L. Evans , Carpenter
co-authored a trivia book titled Matching Wits With the Million - Dollar Mind : The World 's Hardest Trivia Quizzes From America 's First Quiz Show Millionaire
. The book was published by Berkley Books in 2002 . Other game show appearances ( edit ) In 2000 , Carpenter appeared in the Who Wants to Be a Millionaire
Champions Edition , in which previous contestants who won $250,000 -- $1,000,000 played again , with half of their additional winnings from the Champions
Edition going to a charity of their choice . Carpenter won $250,000 , bringing his total Millionaire winnings to $1,250,000 . In 2004 , Carpenter participated
in Super Millionaire , as one of the `` Three Wise Men '' on the episode during which Robert `` Bob - O '' Essig won $1,000,000 . He later participated as part
of the Mob on NBC 's 1 vs.
</item 8>

<item 9>
John Carpenter (game show contestant)
Retrieved September 6 , 2009 . ^ Jump up to : `` Who Wants to Be a Millionaire '' . Gameshow Hall of Fame . GSN . January 21 , 2007 . ^ Jump up to : Vigoda ,
Arlene ( November 22 , 1999 ) . `` Million - dollar winner untaxed by celebrity '' . USA Today . Gannett Company . p. 1D . Jump up ^ `` Taxman scoops a million
'' . BBC News . November 21 , 1999 . Archived from the original on June 21 , 2015 . Retrieved August 1 , 2014 . Jump up ^ Davis , Kristin ( November 2000 ) . ``
I 'm Rich ! ( Now what ? ) '' . Kiplinger 's Personal Finance . pp. 93 -- 94 . ^ Jump up to : Duca , Lauren ( August 15 , 2014 ) . `` The Final Answer On Who
Wants To Be A Millionaire ? , 15 Years After It Premiered '' .
</item 9>

<item 10>
John Carpenter (game show contestant)
John Carpenter ( game show contestant ) - wikipedia John Carpenter ( game show contestant ) Jump to : navigation , search John Carpenter 1968 ( age 48 -- 49 )
Education Rutgers University Occupation IRS agent Known for First top prize winner on Who Wants to Be a Millionaire Home town Northampton , Massachusetts John
Carpenter ( born c. 1968 ) is an American game show contestant and IRS agent . He is best known for becoming the first top prize winner on the United States
version of Who Wants to Be a Millionaire . He held the record for the largest single win in United States game show history , until it was broken by Rahim
Oberholtzer who won $1.12 million on another U.S. quiz show , Twenty One . Carpenter was also the first top prize winner among all international versions of the
Who Wants to Be a Millionaire ? series . On the November 19 , 1999 , episode of Millionaire , Carpenter proceeded to advance to the million - dollar question
without using any lifelines . He then used his Phone - A-Friend to call his father not for help , but rather to tell him he was going to win the game .
</item 10>
=== END ITEM id=253c0c9a7f06aa70 ===

=== ITEM id=ef1f443438ac36b4 ===
Gold sentences and facts:
Sentence 0: Bay of Montevideo
  Fact 0.0: Bay of Montevideo

Retrieved items:
<item 1>
In Colombia, routes connect Medellín and Bogotá, continue south on Highway 25 to Ecuador, and link Bogotá with Venezuela via the Simón Bolívar Highway to
Guiria. Ecuador’s Highway 35 and Peru’s Highway 1 carry the route south. In Chile it branches toward Puerto Montt and Chiloé, or east to Buenos Aires;
extensions reach Ushuaia and Asunción. A ferry and roads connect Buenos Aires with Uruguay and Brazil.
</item 1>

<item 2>
Pan-American Highway
Other branches probably exist across the center of South America . Almost all Pan-American sections in Gran Buenos Aires are modern and fast main highways West
Access to Buenos Aires Brazil and Uruguay ( edit ) A continuation of the Pan-American Highway to the Brazilian cities of São Paulo and Rio de Janeiro uses a
ferry from Buenos Aires to Colonia in Uruguay and Uruguay Highway 1 to Montevideo . Uruguay Highway 9 and Brazil Highway 471 route to near Pelotas , from where
Brazil Highway 116 leads to Brazilian main cities . Guyana , Suriname and French Guyana ( edit ) The highway does not have official segments to Belize , Guyana
, Suriname and French Guiana , nor to any of the island nations in the Americas . However , highways from Venezuela link to Brazilian Trans - Amazonian highway
that provides a southwest entrance to Guyana , route to the coast , and follow a coastal route through Suriname to French Guiana . West Indies section ( edit )
Plans have been discussed for including the West Indies in the Pan American Highway system . According to these , a system of ferries would be established to
connect terminal points of the highway .
</item 2>

<item 3>
The text catalogs seas worldwide, including Pacific and Southern Ocean seas, plus landlocked seas. It outlines the Colorado River system: jurisdictions,
canyons, landscapes, tributaries, dams, reservoirs, aqueducts, canals, water projects, parks, and related agencies, agreements, and figures. It also lists
Southern Border Region communities, highlighting San Diego as a 1.3-million-person global city and naming nearby cities and towns by population.
</item 3>

<item 4>
FIFA World Cup hosts
Under FIFA rules as of 2017 , the 2030 World Cup can not be held in Asia ( AFC ) because the Asian Football Confederation is excluded from the bidding following
the selection of Qatar in 2022 . Also in June 2017 , UEFA 's president Aleksander Čeferin stated that Europe ( UEFA ) will definitely fight for its right to
host the 2030 World Cup . The Uruguay -- Argentina bid would coincide with the centennial anniversary of the first FIFA World Cup final , and the bicentennial
of the first Constitution of Uruguay . The Uruguay - Argentina bid was officially confirmed on 29 July 2017 . Paraguay is aiming for joining the bid , but this
is not confirmed or discarded yet . The efforts for the inclusion of Paraguay in the bid caused discomfort in the Uruguayan Football Association , and Wilmar
Valdéz ( AUF president ) accused some insistence of the Conmebol although it is not official that Paraguay joins the event . Chile also wants to organize the
World Cup with Uruguay and Argentina . A joint bid was announced by the Argentine Football Association and the Uruguayan Football Association on 29 July 2017 .
</item 4>

<item 5>
Uruguay was chosen unopposed to host the inaugural 1930 World Cup after Hungary, the Netherlands, Sweden, Italy and Spain withdrew; the Congress ratified the
choice in Barcelona on 18 May 1929. Montevideo’s Estadio Centenario commemorated Uruguay’s constitutional centennial. Sweden’s withdrawal left Italy as host for
1934. World War II cancelled the planned 1942 Cup. Brazil was selected in 1946 for the 1950 tournament, initially scheduled for
</item 5>

<item 6>
The excerpt contains Wikipedia page metadata for “Uvea,” followed by the category “Bodies of water of Uruguay.” That category has four subcategories—bays,
lakes, rivers, and springs—and lists Bay of Montevideo as its sole page. The pages include standard Wikipedia licensing, editing, and navigation information.
</item 6>

<item 7>
FIFA World Cup hosts
The original bid logo for the Uruguay - Argentina World Cup Bid . Confirmed 2030 bids : CONMEBOL : Uruguay / Argentina / Paraguay : Uruguay -- Argentina --
Paraguay 2030 FIFA World Cup bid Expressed interest in bidding : UEFA : England : England 2030 FIFA World Cup bid 2034 FIFA World Cup ( edit ) Main article :
2034 FIFA World Cup The first bid for the 2034 FIFA World Cup has been proposed as a collective bid by the members of the Association of Southeast Asian Nations
. The idea of a combined ASEAN bid had been mooted as early as January 2011 , when the Football Association of Singapore 's then - President , Zainudin Nordin ,
said in a statement that the proposal had been made at an ASEAN Foreign Ministers meeting . In 2013 , Nordin and Special Olympics Malaysia President , Datuk
Mohamed Feisol Hassan , recalled the idea for ASEAN to jointly host a World Cup . Under FIFA rules as of 2017 , the 2030 World Cup can not be held in Asia ( AFC
) as Asian Football Confederation members are excluded from the bidding following the selection of Qatar in 2022 .
</item 7>

<item 8>
FIFA World Cup hosts
Before Uruguay and Argentina played out a goalless draw in Montevideo , FC Barcelona players Luis Suárez and Lionel Messi promoted the bid with commemorative
shirts . On 31 August 2017 , it was suggested Paraguay would join as a third host . CONMEBOL , the South American confederation , confirmed the joint three -
way bid in September 2017 . English FA vice chairman David Gill has proposed that his country could potentially bid for 2030 , provided that the bidding process
is made more transparent . `` England is one of few countries that could stage even a 48 - nation event in its entirety , while Football Association chief
executive Martin Glenn made it clear earlier this year bidding for 2030 was an option . '' In June 2017 , UEFA stated that `` it would support a pan-British bid
for 2030 or even a single bid from England . '' Moreover , a possible English bid for 2030 was also backed by the German Football Association . It is expected
that countries from `` South America ( CONMEBOL ) and Europe ( UEFA ) will likely to be in a bid competition for 2030 . ''
</item 8>

<item 9>
FIFA World Cup hosts
Retrieved 14 January 2017 . Jump up ^ `` UEFA will ' fight ' for Europe 's right to host 2030 World Cup '' . Retrieved 2 June 2017 . ^ Jump up to : Molestia en
Uruguay por video de Conmebol ( in Spanish ) Jump up ^ ElPais . `` Chile no se rinde con ser sede '' ( in Spanish ) . Retrieved 15 March 2018 . Jump up ^ Dudko
, James ( 29 July 2017 ) . `` Argentina , Uruguay Announce Bid to Host 2030 FIFA World Cup '' . Bleacher Report . Turner Broadcasting System . Jump up ^ Gadd ,
Mick ( 1 September 2017 ) . `` Lionel Messi and Luis Suarez campaign for Uruguay and Argentina 's 2030 World Cup bid before 0 - 0 draw '' . Daily Mirror . Jump
up ^ `` Es oficial : Paraguay se suma a la candidatura de Argentina y Uruguay para el Mundial 2030 '' ( It 's official : Paraguay adds itself to the Argentina
and Uruguay bid ) ( in Spanish ) .
</item 9>

<item 10>
Category:Bodies of water of Uruguay
Category : Bodies of water of Uruguay - wikipedia Help Category : Bodies of water of Uruguay Jump to : navigation , search Wikimedia Commons has media related
to Bodies of water in Uruguay . This category has the following 4 subcategories , out of 4 total . ► Bays of Uruguay ‎ ( 1 P ) ► Lakes of Uruguay ‎ ( 5 P ) ►
Rivers of Uruguay ‎ ( 2 C , 35 P ) ► Springs of Uruguay ‎ ( 1 P ) Pages in category `` Bodies of water of Uruguay '' This category contains only the following
page . This list may not reflect recent changes ( learn more ) . Bay of Montevideo Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Category:Bodies_of_water_of_Uruguay&oldid=756422847 '' Categories : Landforms of Uruguay Bodies of water by country
Water in Uruguay Bodies of water of South America by country Talk Help About Wikipedia Azərbaycanca Deutsch فارسی 한국어 Magyar Português Türkçe اردو Edit links
This page was last edited on 24 December 2016 , at 03 : 52 . About Wikipedia
</item 10>
=== END ITEM id=ef1f443438ac36b4 ===

=== ITEM id=e3d771a678664c67 ===
Gold sentences and facts:
Sentence 0: Brain tissue consumes a large amount of energy in proportion to its volume , so large brains place severe metabolic demands on animals . The need to limit body weight in order , for example , to fly , has apparently led to selection for a reduction of brain size in some species , such as bats . Most of the brain 's energy consumption goes into sustaining the electric charge ( membrane potential ) of neurons . Most vertebrate species devote between 2 % and 8 % of basal metabolism to the brain . In primates , however , the percentage is much higher -- in humans it rises to 20 -- 25 % . The energy consumption of the brain does not vary greatly over time , but active regions of the cerebral cortex consume somewhat more energy than inactive regions ; this forms the basis for the functional brain imaging methods PET , fMRI , and NIRS . The brain typically gets most of its energy from oxygen - dependent metabolism of glucose ( i.e. , blood sugar ) , but ketones provide a major alternative source , together with contributions from medium chain fatty acids ( caprylic and heptanoic acids ) , lactate , acetate , and possibly amino acids .
  Fact 0.0: Brain tissue consumes a large amount of energy in proportion to its volume.
  Fact 0.1: Large brains place severe metabolic demands on animals.
  Fact 0.2: The need to limit body weight, for example in order to fly, has apparently led to selection for a reduction of brain size in some species.
  Fact 0.3: Bats are one of the species that have apparently been selected for reduced brain size.
  Fact 0.4: Most of the brain's energy consumption goes into sustaining the electric charge (membrane potential) of neurons.
  Fact 0.5: Most vertebrate species devote between 2% and 8% of basal metabolism to the brain.
  Fact 0.6: In primates, the percentage of basal metabolism devoted to the brain is much higher.
  Fact 0.7: In humans, the brain's share of basal metabolism rises to 20-25%.
  Fact 0.8: The energy consumption of the brain does not vary greatly over time.
  Fact 0.9: Active regions of the cerebral cortex consume somewhat more energy than inactive regions.
  Fact 0.10: This difference in energy use forms the basis for the functional brain imaging methods PET, fMRI, and NIRS.
  Fact 0.11: The brain typically gets most of its energy from oxygen-dependent metabolism of glucose (blood sugar).
  Fact 0.12: Ketones provide a major alternative energy source for the brain.
  Fact 0.13: Medium chain fatty acids (caprylic and heptanoic acids), lactate, acetate, and possibly amino acids also contribute to the brain's energy supply.

Retrieved items:
<item 1>
Brain
Most of the brain 's energy consumption goes into sustaining the electric charge ( membrane potential ) of neurons . Most vertebrate species devote between 2 %
and 8 % of basal metabolism to the brain . In primates , however , the percentage is much higher -- in humans it rises to 20 -- 25 % . The energy consumption of
the brain does not vary greatly over time , but active regions of the cerebral cortex consume somewhat more energy than inactive regions ; this forms the basis
for the functional brain imaging methods PET , fMRI , and NIRS . The brain typically gets most of its energy from oxygen - dependent metabolism of glucose (
i.e. , blood sugar ) , but ketones provide a major alternative source , together with contributions from medium chain fatty acids ( caprylic and heptanoic acids
) , lactate , acetate , and possibly amino acids . Functions Model of a neural circuit in the cerebellum , as proposed by James S. Albus . Information from the
sense organs is collected in the brain . There it is used to determine what actions the organism is to take . The brain processes the raw data to extract
information about the structure of the environment .
</item 1>

<item 2>
Brain
In rituals Some archaeological evidence suggests that the mourning rituals of European Neanderthals also involved the consumption of the brain . The Fore people
of Papua New Guinea are known to eat human brains . In funerary rituals , those close to the dead would eat the brain of the deceased to create a sense of
immortality . A prion disease called kuru has been traced to this . For Tanning The brain can be useful to hunters : most animals have enough brain matter for
use in the tanning of their own hides . See also Brain -- computer interface Central nervous system disease List of neuroscience databases Neurological disorder
Neuroplasticity Optogenetics Outline of neuroscience References Jump up ^ Pelvig , DP ; Pakkenberg , H ; Stark , AK ; Pakkenberg , B ( 2008 ) . `` Neocortical
glial cell numbers in human brains '' . Neurobiology of Aging. 29 ( 11 ) : 1754 -- 1762 . doi : 10.1016 / j. neurobiolaging. 2007.04. 013 . PMID 17544173 . Jump
up ^ Yuste , Rafael ; Church , George M.
</item 2>

<item 3>
Brain
Jump up ^ Armstrong , E ( 1983 ) . `` Relative brain size and metabolism in mammals '' . Science . 220 ( 4603 ) : 1302 -- 1304 . Bibcode : 1983Sci ... 220.1302
A . doi : 10.1126 / science. 6407108 . PMID 6407108 . Jump up ^ Jerison , HJ ( 1973 ) . Evolution of the Brain and Intelligence . Academic Press . pp. 55 -- 74
. ISBN 978 - 0 - 12 - 385250 - 2 . Jump up ^ Parent , A ; Carpenter , MB ( 1995 ) . `` Ch. 1 '' . Carpenter 's Human Neuroanatomy . Williams & Wilkins . ISBN
978 - 0 - 683 - 06752 - 1 . Jump up ^ Pardridge , W ( 2005 ) . `` The Blood - Brain Barrier : Bottleneck in Brain Drug Development '' . NeuroRx. 2 ( 1 ) : 3 --
14 .
</item 3>

<item 4>
Brain
Jump up ^ Rang , HP ( 2003 ) . Pharmacology . Churchill Livingstone . pp. 476 -- 483 . ISBN 0 - 443 - 07145 - 4 . Jump up ^ Speckmann , E-J ; Elger , CE ( 2004
) . `` Introduction to the neurophysiological basis of the EEG and DC potentials '' . In Niedermeyer E , Lopes da Silva FH . Electroencephalography : Basic
Principles , Clinical Applications , and Related Fields . Lippincott Williams & Wilkins . pp. 17 -- 31 . ISBN 0 - 7817 - 5126 - 8 . CS1 maint : Uses editors
parameter ( link ) ^ Jump up to : Buzsáki , G ( 2006 ) . Rhythms of the Brain . Oxford University Press . ISBN 978 - 0 - 19 - 530106 - 9 . OCLC 63279497 .
</item 4>

<item 5>
Brain
PMID 7282965 . Jump up ^ Raichle , M ; Gusnard , DA ( 2002 ) . `` Appraising the brain 's energy budget '' . Proc. Natl. Acad. Sci. U.S.A. 99 ( 16 ) : 10237 --
10239 . Bibcode : 2002PNAS ... 9910237R . doi : 10.1073 / pnas. 172399499 . PMC 124895 . PMID 12149485 . Jump up ^ Mehagnoul - Schipper , DJ ; Van Der Kallen ,
BF ; Colier , WNJM ; Van Der Sluijs , MC ; Van Erning , LJ ; Thijssen , HO ; Oeseburg , B ; Hoefnagels , WH ; Jansen , RW ( 2002 ) . `` Simultaneous
measurements of cerebral oxygenation changes during brain activation by near - infrared spectroscopy and functional magnetic resonance imaging in healthy young
and elderly subjects '' . Hum Brain Mapp. 16 ( 1 ) : 14 -- 23 .
</item 5>

<item 6>
Brain
2012.151 . PMC 3564188 . PMID 23072752 . Jump up ^ Boumezbeur , F. ; Petersen , KF. ; Cline , GW. ; Mason , GF. ; Behar , KL. ; Shulman , GI. ; Rothman , DL. (
Oct 2010 ) . `` The contribution of blood lactate to brain energy metabolism in humans measured by dynamic 13C nuclear magnetic resonance spectroscopy '' . J
Neurosci. 30 ( 42 ) : 13983 -- 91 . doi : 10.1523 / JNEUROSCI. 2040 - 10.2010 . PMC 2996729 . PMID 20962220 . Jump up ^ Deelchand , DK. ; Shestov , AA. ; Koski
, DM. ; Uğurbil , K. ; Henry , PG . ( May 2009 ) . `` Acetate transport and utilization in the rat brain '' . J Neurochem.
</item 6>

<item 7>
Brain
`` Acetate transport and utilization in the rat brain '' . J Neurochem. 109 Suppl 1 ( Suppl 1 ) : 46 -- 54 . doi : 10.1111 / j. 1471 - 4159.2009. 05895. x . PMC
2722917 . PMID 19393008 . Jump up ^ Soengas , JL ; Aldegunde , M ( 2002 ) . `` Energy metabolism of fish brain '' . Comparative Biochemistry and Physiology B.
131 ( 3 ) : 271 -- 296 . doi : 10.1016 / S1096 - 4959 ( 02 ) 00022 - 2 . PMID 11959012 . ^ Jump up to : Carew , TJ ( 2000 ) . `` Ch. 1 '' . Behavioral
Neurobiology : the Cellular Organization of Natural Behavior . Sinauer Associates . ISBN 978 - 0 - 87893 - 092 - 0 . Jump up ^ Dafny , N. `` Anatomy of the
spinal cord '' .
</item 7>

<item 8>
Brain
Hum Brain Mapp. 16 ( 1 ) : 14 -- 23 . doi : 10.1002 / hbm. 10026 . PMID 11870923 . Jump up ^ Ebert , D. ; Haller , RG. ; Walton , ME . ( Jul 2003 ) . `` Energy
contribution of octanoate to intact rat brain metabolism measured by 13C nuclear magnetic resonance spectroscopy '' . J Neurosci. 23 ( 13 ) : 5928 -- 35 . PMID
12843297 . Jump up ^ Marin - Valencia , I. ; Good , LB. ; Ma , Q. ; Malloy , CR. ; Pascual , JM . ( Feb 2013 ) . `` Heptanoate as a neural fuel : energetic and
neurotransmitter precursors in normal and glucose transporter I - deficient ( G1D ) brain '' . J Cereb Blood Flow Metab. 33 ( 2 ) : 175 -- 82 . doi : 10.1038 /
jcbfm. 2012.151 . PMC 3564188 .
</item 8>

<item 9>
Brain
Norepinephrine , which is involved in arousal , comes exclusively from a nearby small area called the locus coeruleus . Other neurotransmitters such as
acetylcholine and dopamine have multiple sources in the brain , but are not as ubiquitously distributed as glutamate and GABA . Electrical activity Brain
electrical activity recorded from a human patient during an epileptic seizure . As a side effect of the electrochemical processes used by neurons for signaling
, brain tissue generates electric fields when it is active . When large numbers of neurons show synchronized activity , the electric fields that they generate
can be large enough to detect outside the skull , using electroencephalography ( EEG ) or magnetoencephalography ( MEG ) . EEG recordings , along with
recordings made from electrodes implanted inside the brains of animals such as rats , show that the brain of a living animal is constantly active , even during
sleep . Each part of the brain shows a mixture of rhythmic and nonrhythmic activity , which may vary according to behavioral state .
</item 9>

<item 10>
Brain
In mammals , the cerebral cortex tends to show large slow delta waves during sleep , faster alpha waves when the animal is awake but inattentive , and chaotic -
looking irregular activity when the animal is actively engaged in a task . During an epileptic seizure , the brain 's inhibitory control mechanisms fail to
function and electrical activity rises to pathological levels , producing EEG traces that show large wave and spike patterns not seen in a healthy brain .
Relating these population - level patterns to the computational functions of individual neurons is a major focus of current research in neurophysiology .
Metabolism All vertebrates have a blood -- brain barrier that allows metabolism inside the brain to operate differently from metabolism in other parts of the
body . Glial cells play a major role in brain metabolism by controlling the chemical composition of the fluid that surrounds neurons , including levels of ions
and nutrients . Brain tissue consumes a large amount of energy in proportion to its volume , so large brains place severe metabolic demands on animals . The
need to limit body weight in order , for example , to fly , has apparently led to selection for a reduction of brain size in some species , such as bats .
</item 10>
=== END ITEM id=e3d771a678664c67 ===
